# core/llm_router.py
from dataclasses import dataclass
from typing import List, Dict, Tuple, Callable
from core.qwen_api import chat_qwen

@dataclass
class RouteDecision:
    use_local_lora: bool
    reason: str

def decide_route(user_text: str, stage: int) -> RouteDecision:
    emo_kw = ["难过","委屈","想哭","好累","压力","失眠","孤独","emo","烦","想你","抱抱","晚安","早安"]
    t = (user_text or "")
    is_emo = any(k in t for k in emo_kw)
    if is_emo or int(stage) >= 1:
        return RouteDecision(True, "emotional_or_relationship")
    return RouteDecision(False, "general_or_task")

def routed_chat(
    messages: List[Dict],
    user_text: str,
    stage: int,
    local_chat_fn: Callable[[List[Dict], float], str],
    temperature: float = 0.7,
    max_tokens: int = 256
) -> Tuple[str, RouteDecision]:
    d = decide_route(user_text, stage)
    if d.use_local_lora:
        return local_chat_fn(messages, temperature), d
    return chat_qwen(messages, temperature=temperature, max_tokens=max_tokens), d