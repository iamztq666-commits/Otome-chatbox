# core/llm_router.py
from dataclasses import dataclass
from typing import List, Dict, Tuple, Callable
from core.qwen_api import chat_qwen

@dataclass
class RouteDecision:
    use_local_lora: bool
    reason: str

def decide_route(user_text: str, stage: int) -> RouteDecision:
    # 暂时全走 Qwen API，等 LoRA 训练好再开启本地路由
    # 之后只需把下面这行改回来：
    # emo_kw = ["难过","委屈","想哭","好累","压力","失眠","孤独","emo","烦","想你","抱抱","晚安","早安"]
    # is_emo = any(k in (user_text or "") for k in emo_kw)
    # if is_emo or int(stage) >= 1:
    #     return RouteDecision(True, "emotional_or_relationship")
    return RouteDecision(False, "api_only")

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
        result = local_chat_fn(messages, temperature)
        if result is not None:
            return result, d
        # 本地模型不可用时 fallback 到 API
    return chat_qwen(messages, temperature=temperature, max_tokens=max_tokens), d
