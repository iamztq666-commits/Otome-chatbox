# src/core/qwen_api.py
import os
from openai import OpenAI

_client = None

def _get_client() -> OpenAI:
    global _client
    if _client is not None:
        return _client

    api_key = os.getenv("QWEN_API_KEY")
    base_url = os.getenv("QWEN_BASE_URL")
    if not api_key or not base_url:
        raise RuntimeError("Qwen API not configured: set QWEN_API_KEY and QWEN_BASE_URL")

    _client = OpenAI(api_key=api_key, base_url=base_url)
    return _client

def chat_qwen(messages, temperature=0.7, max_tokens=256):
    client = _get_client()
    model = os.getenv("QWEN_MODEL", "qwen-plus")
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()