import os
import gradio as gr

from core.llm import load_llm, chat
from core.prompt import build_prompt
from core.state import load_state, save_state, update_trust_stage
from rag.retrieve import retrieve_canon
from story.unlock import unlock_new_plots

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
LORA_PATH = os.getenv("LORA_PATH")  # 可选：./lora_idol

tokenizer, model = load_llm(MODEL_NAME, lora_path=LORA_PATH)

persona = open("./data/canon/persona.md", "r", encoding="utf-8").read()

def stage_label(stage: int) -> str:
    return ["初识", "常驻支持", "核心应援", "团队同盟"][max(0, min(stage, 3))]

def respond(user_msg, history, trust_box, stage_box, plots_box):
    user_id = "u1"
    history = history or []

    state = load_state(user_id)
    # 更新信任（基于用户输入的即时规则）
    state = update_trust_stage(state, user_msg)

    # RAG 检索：where gating（stage/trust/spoiler）
    canon_pairs = retrieve_canon(
        query=user_msg,
        stage=state.stage,
        trust=state.trust,
        spoiler_level=state.spoiler_level,
        k=6
    )

    prompt = build_prompt(
        persona=persona,
        state={"trust": state.trust, "stage": state.stage, "spoiler_level": state.spoiler_level},
        canon_pairs=canon_pairs,
        history=history,
        user_msg=user_msg,
    )

    reply = chat(tokenizer, model, prompt, temperature=0.7)

    # 解锁剧情：基于本轮检索到的 plot meta（更稳、天然符合where gating）
    newly = unlock_new_plots(state, canon_pairs)
    if newly:
        for x in newly:
            state.unlocked_plots.append(x["plot_id"])
        unlock_text = "🎬 解锁剧情：\n" + "\n".join([f"- {x['title']}（{x['plot_id']}）" for x in newly])
        reply = reply + "\n\n" + unlock_text

    save_state(user_id, state)

    history.append((user_msg, reply))

    trust_box = f"{state.trust}/100"
    stage_box = f"{stage_label(state.stage)}（stage={state.stage}, spoiler={state.spoiler_level}）"
    plots_box = "\n".join([f"- {pid}" for pid in state.unlocked_plots]) or "（暂无）"

    return history, "", trust_box, stage_box, plots_box

with gr.Blocks() as demo:
    gr.Markdown("# 爱豆养成系统 Demo（RAG where gating + 信任阶段 + 剧情解锁 + 营业CP/竞争边界）")

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=560)
            msg = gr.Textbox(label="输入", placeholder="比如：今天练声打卡了吗？/你和周衡是不是营业？")
        with gr.Column(scale=1):
            trust_box = gr.Textbox(label="信任值", value="0/100", interactive=False)
            stage_box = gr.Textbox(label="阶段", value="初识", interactive=False)
            plots_box = gr.Textbox(label="已解锁剧情", value="（暂无）", lines=12, interactive=False)
            gr.Markdown("提示：信任越高，where 会放行更多 plot / 内部心态内容。")

    msg.submit(respond, [msg, chatbot, trust_box, stage_box, plots_box],
               [chatbot, msg, trust_box, stage_box, plots_box])

demo.launch(server_name="0.0.0.0", server_port=7860)


