import os
from typing import Dict, List, Tuple

LETTA_BASE_URL = os.getenv("LETTA_BASE_URL", "http://127.0.0.1:8283")
LETTA_API_KEY = os.getenv("LETTA_API_KEY")  # 自托管可不填；云端需要


def _get_client():
    # 两种常见导入名，哪个能用用哪个
    try:
        from letta_client import Letta  # 常见
        return Letta(apiKey=LETTA_API_KEY, baseURL=LETTA_BASE_URL)
    except Exception:
        from letta_ai_letta_client import Letta  # 兜底（不同打包名时）
        return Letta(apiKey=LETTA_API_KEY, baseURL=LETTA_BASE_URL)


def ensure_agent(agent_name: str, persona_text: str) -> str:
    """
    创建（或复用）一个 Letta agent，返回 agent_id
    乙游化：human memory block 变成“用户是恋爱对话对象/慢热陪伴”而非“训练监督搭子”
    """
    client = _get_client()

    # 1) 列表里找同名 agent
    agents = client.agents.list()
    for a in agents:
        if a.get("name") == agent_name:
            return a.get("id")

    # 2) 创建 agent：用 memory_blocks 填 persona/human
    agent = client.agents.create({
        "name": agent_name,
        "memory_blocks": [
            {"label": "persona", "value": persona_text},
            {
                "label": "human",
                "value": (
                    "用户是你在乙游世界里最重要、最在意的人。你与用户以“慢热、健康、尊重边界”的恋爱关系推进。"
                    "你温柔细腻、擅长共情与陪伴，会在日常对话中表达在意与偏爱，但不操控、不逼迫、不越界。"
                )
            },
        ],
    })
    return agent.get("id")


def chat_with_agent(agent_id: str, user_msg: str, canon_block: str, state_block: str) -> str:
    """
    每轮对话：把 CANON + STATE 注入为 system context，再发送 user 消息
    乙游化：把“公开口径/营业CP/行程机密”等改成“慢热、边界、安全、事实基于 canon”
    """
    client = _get_client()

    # 乙游 system 注入
    system_injection = f"""
[RELATIONSHIP_STATE]
{state_block}

[CANON_FACTS]
{canon_block}

[ROLE_AND_GOAL]
你在扮演一位乙游男主/虚拟恋人向角色。目标是提供“情绪陪伴 + 关系推进”的沉浸式对话：
温柔、细腻、会倾听与共情；推进关系要慢热自然，不油腻、不强行升温。

[HARD_CONSTRAINTS]
- 世界观/人设/剧情事实优先基于 CANON_FACTS；不确定就用“我记得大概是…/我想再确认一下”或向用户反问，不要编造硬细节。
- 关系推进遵循 stage：stage 0-1 仅轻微暧昧；stage 2 可表达心动；stage 3 才能表达稳定关系，但依然克制含蓄。
- 尊重用户边界：不PUA、不威胁、不羞辱、不道德绑架（如“你不回我就…”）。
- 禁止露骨性描写与明确性行为细节；亲密描写以含蓄方式为主，并以用户意愿为准。
- 遇到现实危机（自伤、违法等）要劝其寻求现实帮助并保持关怀语气。
- 输出风格：优先对话感，尽量包含 1句情绪回应 + 1句关心追问 + 1个轻互动（小约定/小玩笑/选择题）。
""".strip()

    resp = client.agents.messages.create(agent_id, {
        "messages": [
            {"role": "system", "content": system_injection},
            {"role": "user", "content": (user_msg or "").strip()},
        ]
    })

    # resp 结构随 SDK 版本不同，做稳健提取：找到 assistant 最后一条
    msgs = None
    if isinstance(resp, dict):
        msgs = resp.get("messages") or resp.get("data")
    if msgs is None:
        msgs = []

    if isinstance(msgs, list):
        for m in reversed(msgs):
            if isinstance(m, dict) and m.get("role") == "assistant":
                content = m.get("content", "")
                return content.strip() if isinstance(content, str) else str(content).strip()

    # 兜底
    return str(resp).strip()
