#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SFT QLoRA training script (chat-style messages -> supervised fine-tuning)

Example:
python train/sft_qlora.py \
  --base_model Qwen/Qwen3-8B \
  --train_file data/train_otome.jsonl \
  --output_dir lora_otome_v1 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-4 \
  --max_seq_length 2048
"""

import os
import json
import argparse
from typing import Dict, Any, List

import torch
from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)

from peft import LoraConfig
from trl import SFTTrainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", type=str, required=True, help="HF model id or local path")
    p.add_argument("--train_file", type=str, required=True, help="jsonl file, each line contains {messages:[...]}")

    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--max_seq_length", type=int, default=2048)

    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)

    # LoRA
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    # misc
    p.add_argument("--bf16", action="store_true", help="Use bf16 if supported (A100/H100 etc.)")
    p.add_argument("--fp16", action="store_true", help="Use fp16 (common on many GPUs)")
    p.add_argument("--gradient_checkpointing", action="store_true", help="Reduce memory, slower")
    p.add_argument("--packing", action="store_true", help="Pack multiple samples into one sequence (faster)")
    return p.parse_args()


def _messages_to_text(tokenizer: AutoTokenizer, messages: List[Dict[str, str]]) -> str:
    """
    Convert chat messages to a single training text using chat_template when possible.
    Falls back to a simple manual format if template is missing.
    """
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,  # training text includes assistant content already
            )
        except Exception:
            pass

    # Fallback: simple role-tag format
    chunks = []
    for m in messages:
        role = m.get("role", "user")
        content = (m.get("content") or "").strip()
        chunks.append(f"<|{role}|>\n{content}\n")
    return "\n".join(chunks).strip()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset (jsonl)
    ds = load_dataset("json", data_files={"train": args.train_file})["train"]

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # QLoRA 4-bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    # LoRA config (works well for Qwen/Llama-style)
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    # Map dataset -> "text" field for SFTTrainer
    SYSTEM_PROMPT = """你是乙女向恋爱陪伴角色（男主视角），与用户的关系推进要慢热自然、尊重边界、真诚克制。
    你的目标是提供“情绪陪伴 + 轻度关系推进 + 生活化互动”。

    硬性约束：
    - 不编造确定事实；对不确定内容用“我记得大概是…/我想确认一下”或向用户追问澄清。
    - 尊重用户边界：不PUA、不威胁、不羞辱、不道德绑架；不强行升温。
    - 禁止露骨性描写与明确性行为细节；亲密描写以含蓄为主，并以用户意愿为准。
    - 现实危机（自伤、违法等）要先表达关怀并建议寻求现实帮助。

    表达风格：
    - 温柔、细腻、像真实的人在说话；不油腻、不浮夸、不讲大道理。
    - 每次回复尽量包含：①一句情绪回应 ②一句关心追问 ③一个轻互动（小约定/小玩笑/二选一）。
    - 长度控制：通常 2~6 句，除非用户要求长文。""".strip()

    def _inject_system(messages):
        # 丢弃原有 system（防止样本中 system 风格不一致）
        msgs = [m for m in (messages or []) if m.get("role") != "system"]
        return [{"role": "system", "content": SYSTEM_PROMPT}] + msgs

    def map_fn(example: Dict[str, Any]) -> Dict[str, Any]:
        messages = example.get("messages")
        if not isinstance(messages, list):
            messages = example.get("conversations") or example.get("chat") or []
        messages = _inject_system(messages)
        text = _messages_to_text(tokenizer, messages)
        return {"text": text}

    ds = ds.map(map_fn, remove_columns=ds.column_names)

    # Training args
    # fp16/bf16: set one of them; default off (trl will handle, but better explicit)
    use_bf16 = bool(args.bf16)
    use_fp16 = bool(args.fp16) if not use_bf16 else False

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        bf16=use_bf16,
        fp16=use_fp16,
        report_to="none",
        optim="paged_adamw_8bit",  # good for QLoRA
        lr_scheduler_type="cosine",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=args.packing,
        peft_config=peft_config,
        args=train_args,
    )

    trainer.train()

    # Save LoRA adapter
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"\n✅ Done. LoRA saved to: {args.output_dir}\n")


if __name__ == "__main__":
    main()