# ---- workaround: allow boolean JSON Schema (True/False) in gradio_client ----
import gradio_client.utils as gcu

_old_json = gcu._json_schema_to_python_type

def _patched_json_schema_to_python_type(schema, defs=None):
    # JSON Schema allows boolean schemas: True means "any", False means "never"
    if isinstance(schema, bool):
        return "Any" if schema else "Never"
    return _old_json(schema, defs)

gcu._json_schema_to_python_type = _patched_json_schema_to_python_type
# ---------------------------------------------------------------------------



MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
LORA_PATH = os.getenv("LORA_PATH")  # 可选：./lora_otome

tokenizer, model = load_llm(MODEL_NAME, lora_path=LORA_PATH)

# persona 建议放到 data/canon/persona/xxx.md 中，但你目前是单文件也能用
# 如果你已经换成 data/canon/persona/*.md，这里也可以读一个主文件作为 persona_text
persona = open("./data/canon/persona.md", "r", encoding="utf-8").read()


def stage_label(stage: int) -> str:
    return ["初见", "熟悉", "心动", "默契恋人"][max(0, min(int(stage), 3))]


def respond(user_msg, history, affection_box, stage_box, memories_box):
    user_id = "u1"
    history = history or []
    user_msg = (user_msg or "").strip()
    if not user_msg:
        return history, "", affection_box, stage_box, memories_box

    # 1) 先读旧 state（用旧状态来生成本轮回复更自然）
    state = load_state(user_id)

    # 兼容：有些实现把字段叫 trust/stage/memory_unlock
    affection = getattr(state, "trust", 20)
    stage = getattr(state, "stage", 0)
    memory_unlock = getattr(state, "memory_unlock", getattr(state, "spoiler_level", 0))

    # 2) RAG 检索：按乙游门槛过滤
    canon_pairs = retrieve_for_otome(
        query=user_msg,
        stage=stage,
        affection=affection,
        memory_unlock=memory_unlock,
        k=6
    )

    # 3) 构建 prompt（乙游版 build_prompt）
    prompt = build_prompt(
        persona=persona,
        state={
            "trust": affection,          # 兼容你 prompt 里还用 trust 展示 affection
            "stage": stage,
            "tone": getattr(state, "tone", "gentle"),
            "memory_unlock": memory_unlock,
        },
        canon_pairs=canon_pairs,
        history=history,
        user_msg=user_msg,
    )

    reply = chat(tokenizer, model, prompt, temperature=0.7)

    # 4) 本轮结束后更新亲密度/阶段（乙游规则）
    state = update_trust_stage(state, user_msg)

    # 5) 解锁回忆/剧情（可选）
    newly_text = ""
    if unlock_memories is not None:
        state, newly = unlock_memories(state, canon_pairs)
        if newly:
            # 你可以选择“提示解锁”或“悄悄解锁”
            newly_text = "🔓 解锁回忆：\n" + "\n".join([f"- {mid}" for mid in newly])

    if newly_text:
        reply = reply + "\n\n" + newly_text

    # 6) 保存 state + 更新 UI
    save_state(user_id, state)
    history.append((user_msg, reply))

    affection_box = f"{getattr(state, 'trust', 0)}/100"
    stage_box = f"{stage_label(getattr(state, 'stage', 0))}（stage={getattr(state, 'stage', 0)}, memory_unlock={getattr(state, 'memory_unlock', 0)}）"

    unlocked = getattr(state, "unlocked_plots", []) or []
    memories_box = "\n".join([f"- {x}" for x in unlocked]) or "（暂无）"

    return history, "", affection_box, stage_box, memories_box


with gr.Blocks() as demo:
    gr.Markdown("# 乙游对话 Demo（RAG + 亲密度阶段 + 回忆解锁）")

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=560)
            msg = gr.Textbox(
                label="输入",
                placeholder="比如：我今天有点累… / 你会不会一直陪着我？ / 我们下次想去哪里？"
            )
        with gr.Column(scale=1):
            affection_box = gr.Textbox(label="亲密度", value="20/100", interactive=False)
            stage_box = gr.Textbox(label="关系阶段", value="初见", interactive=False)
            memories_box = gr.Textbox(label="已解锁回忆", value="（暂无）", lines=12, interactive=False)
            gr.Markdown("提示：亲密度越高，会解锁更多回忆与剧情片段（通过检索门槛控制）。")

    msg.submit(
        respond,
        [msg, chatbot, affection_box, stage_box, memories_box],
        [chatbot, msg, affection_box, stage_box, memories_box]
    )

demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
