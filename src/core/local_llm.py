# src/core/local_llm.py
import os
from your_llm_loader import load_llm, chat  # 按你实际 import 路径改

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
LORA_PATH = os.getenv("LORA_PATH")  # 乙女 LoRA

_tokenizer = None
_model = None

def _ensure_loaded():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        _tokenizer, _model = load_llm(MODEL_NAME, lora_path=LORA_PATH)
    return _tokenizer, _model

def chat_local_otome(messages, temperature=0.7, max_tokens=256):
    tokenizer, model = _ensure_loaded()
    return chat(tokenizer, model, messages, temperature=temperature)