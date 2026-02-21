def build_messages(persona, state, canon_pairs, history, user_msg):
    system_msg = f"""
你在扮演一位乙游男主/虚拟恋人向角色。你的目标是陪伴式恋爱对话：
温柔、细腻、会倾听与共情，推进关系慢热自然。

[PERSONA]
{persona}

[RELATIONSHIP_STATE]
affection={state.get("trust", 0)}/100
stage={state.get("stage", 0)}

[HARD_CONSTRAINTS]
- 只输出对用户的回复正文，不要输出“User:”等标签，不要输出#标签。
- 不要编造 CANON_FACTS 以外的具体剧情。
"""

    messages = [
        {"role": "system", "content": system_msg}
    ]

    for u, a in history[-6:]:
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})

    messages.append({"role": "user", "content": user_msg})
    return messages
