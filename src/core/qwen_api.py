import os
from openai import OpenAI

QWEN_API_KEY = os.getenv("QWEN_API_KEY")
QWEN_BASE_URL = os.getenv("QWEN_BASE_URL")  # 百炼/通义提供的 OpenAI 兼容 base url
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-plus")  # 例：qwen-plus / qwen-max / qwen-turbo

client = OpenAI(
    api_key=QWEN_API_KEY,
    base_url=QWEN_BASE_URL,
)

def chat_qwen(messages, temperature=0.7, max_tokens=256):
    resp = client.chat.completions.create(
        model=QWEN_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()
