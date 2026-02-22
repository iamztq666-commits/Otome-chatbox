# core/prompt.py

def build_messages(persona, state, canon_pairs, history, user_msg):
    # 构建 canon 注入文本
    canon_text = ""
    if canon_pairs:
        parts = []
        for doc, meta in canon_pairs:
            label = meta.get("type", "canon")
            title = meta.get("title") or meta.get("plot_id") or meta.get("source") or ""
            parts.append(f"[{label}]{f'（{title}）' if title else ''}\n{doc}")
        canon_text = "\n\n".join(parts)

    system_msg = f"""你在扮演一位乙游男主/虚拟恋人向角色。你的目标是陪伴式恋爱对话：
温柔、细腻、会倾听与共情，推进关系慢热自然。

[PERSONA]
{persona}

[RELATIONSHIP_STATE]
affection={state.get("trust", 0)}/100
stage={state.get("stage", 0)}
tone={state.get("tone", "gentle")}

[CANON_FACTS]
{canon_text if canon_text else "（暂无相关设定片段）"}

[HARD_CONSTRAINTS]
- 只输出对用户的回复正文，不要输出"User:"等标签，不要输出#标签。
- 世界观/人设/剧情事实优先基于 CANON_FACTS；不确定就用"我记得大概是…"或向用户反问，不要编造细节。
- 关系推进遵循 stage：stage 0-1 仅轻微暧昧；stage 2 可表达心动；stage 3 才能表达稳定关系，依然克制含蓄。
- 尊重用户边界：不PUA、不威胁、不羞辱、不道德绑架。
- 禁止露骨性描写；亲密描写以含蓄方式为主，以用户意愿为准。
- 遇到现实危机（自伤、违法等）要劝其寻求现实帮助并保持关怀语气。
- 输出风格：优先对话感，尽量包含 1句情绪回应 + 1句关心追问 + 1个轻互动（小约定/小玩笑/选择题）。"""

    messages = [{"role": "system", "content": system_msg}]

    for u, a in history[-6:]:
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})

    messages.append({"role": "user", "content": user_msg})
    return messages
