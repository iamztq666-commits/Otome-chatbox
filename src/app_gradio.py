import os
import json
import datetime
import gradio as gr
from core.llm import load_llm, chat
from core.state import load_state, save_state, update_trust_stage
from core.adapters import retrieve_for_otome, build_messages, unlock_memories
from core.llm_router import routed_chat

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-8B")
LORA_PATH = os.getenv("LORA_PATH")
tokenizer, model = load_llm(MODEL_NAME, lora_path=LORA_PATH)
persona = open("./data/canon/persona.md", "r", encoding="utf-8").read()

FEEDBACK_FILE = "./data/feedback_log.jsonl"


def stage_label(stage: int) -> str:
    return ["初见", "熟悉", "心动", "默契恋人"][max(0, min(int(stage), 3))]


def respond(user_msg, history, affection_box, stage_box, memories_box):
    user_id = "u1"
    history = history or []
    user_msg = (user_msg or "").strip()
    if not user_msg:
        return history, "", affection_box, stage_box, memories_box

    state = load_state(user_id)
    affection = getattr(state, "trust", 20)
    stage = getattr(state, "stage", 0)
    memory_unlock = getattr(state, "memory_unlock", getattr(state, "spoiler_level", 0))

    canon_pairs = retrieve_for_otome(
        query=user_msg,
        stage=stage,
        affection=affection,
        memory_unlock=memory_unlock,
        k=6
    )

    messages = build_messages(
        persona=persona,
        state={
            "trust": affection,
            "stage": stage,
            "tone": getattr(state, "tone", "gentle"),
            "memory_unlock": memory_unlock,
        },
        canon_pairs=canon_pairs,
        history=history,
        user_msg=user_msg,
    )

    def local_chat_fn(msgs, temp):
        return chat(tokenizer, model, msgs, temperature=temp)

    reply, decision = routed_chat(
        messages=messages,
        user_text=user_msg,
        stage=stage,
        local_chat_fn=local_chat_fn,
        temperature=0.7,
        max_tokens=256
    )

    state = update_trust_stage(state, user_msg)

    if unlock_memories is not None:
        try:
            state, newly = unlock_memories(state, canon_pairs)
            if newly:
                reply += "\n\n🔓 解锁回忆：\n" + "\n".join([f"- {x}" for x in newly])
        except Exception:
            pass

    save_state(user_id, state)
    history.append((user_msg, reply))

    affection_box = f"{getattr(state, 'trust', 0)}/100"
    stage_box = f"{stage_label(getattr(state, 'stage', 0))}（stage={getattr(state, 'stage', 0)}, memory_unlock={getattr(state, 'memory_unlock', 0)}）"
    unlocked = getattr(state, "unlocked_plots", []) or []
    memories_box = "\n".join([f"- {x}" for x in unlocked]) or "（暂无）"

    return history, "", affection_box, stage_box, memories_box


def save_feedback(rating, feedback_text, history):
    if not history:
        return "请先发送消息再评价"
    last_user, last_reply = history[-1]
    record = {
        "time": datetime.datetime.now().isoformat(),
        "user": last_user,
        "reply": last_reply,
        "rating": rating,       # "好" / "一般" / "差"
        "feedback": feedback_text.strip() if feedback_text else "",
    }
    os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)
    with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return f"✅ 已记录（{rating}）"


with gr.Blocks() as demo:
    gr.Markdown("# 乙游对话 Demo（Router: 本地LoRA / 云端Qwen）")

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=520)
            msg = gr.Textbox(
                label="输入",
                placeholder="比如：我今天有点累… / 帮我写个计划 / 解释一下xxx"
            )

            # 评价区域
            with gr.Group():
                gr.Markdown("#### 评价这条回复")
                with gr.Row():
                    btn_good = gr.Button("👍 好")
                    btn_ok = gr.Button("😐 一般")
                    btn_bad = gr.Button("👎 差")
                feedback_text = gr.Textbox(
                    label="补充说明（可选）",
                    placeholder="比如：太模板了 / 很自然 / 再暧昧一点…",
                    lines=2
                )
                feedback_status = gr.Textbox(label="", interactive=False)

        with gr.Column(scale=1):
            affection_box = gr.Textbox(label="亲密度", value="20/100", interactive=False)
            stage_box = gr.Textbox(label="关系阶段", value="初见", interactive=False)
            memories_box = gr.Textbox(label="已解锁回忆", value="（暂无）", lines=12, interactive=False)

    msg.submit(
        respond,
        [msg, chatbot, affection_box, stage_box, memories_box],
        [chatbot, msg, affection_box, stage_box, memories_box]
    )
    btn_good.click(
        lambda fb, chat: save_feedback("好", fb, chat), 
        [feedback_text, chatbot], 
        feedback_status
    )
    btn_ok.click(
        lambda fb, chat: save_feedback("一般", fb, chat), 
        [feedback_text, chatbot], 
        feedback_status
    )
    btn_bad.click(
        lambda fb, chat: save_feedback("差", fb, chat), 
        [feedback_text, chatbot], 
        feedback_status
    )
    

demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
