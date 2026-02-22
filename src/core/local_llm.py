# src/core/local_llm.py
import os
from core.llm import load_llm, chat

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-8B")
LORA_PATH = os.getenv("LORA_PATH")  # 乙女 LoRA，没有就留空

_tokenizer = None
_model = None

def _ensure_loaded():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        _tokenizer, _model = load_llm(MODEL_NAME, lora_path=LORA_PATH)
    return _tokenizer, _model

def chat_local_otome(messages, temperature=0.7, max_tokens=256):
    try:
        tokenizer, model = _ensure_loaded()
        return chat(tokenizer, model, messages, temperature=temperature)
    except Exception as e:
        print(f"[local_llm] 本地模型加载失败，fallback 到 API: {e}")
        return None  # 返回 None 让 router fallback 到 Qwen API
