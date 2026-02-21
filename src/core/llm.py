import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_llm(model_name: str, lora_path: str | None = None):
    tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    use_fast=False,  
)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",            # 放GPU
        torch_dtype=torch.float16,    # GPU fp16
        trust_remote_code=True,
    )
    if lora_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_path)
        # 不 merge：方便以后热切换不同LoRA
    model.eval()
    return tokenizer, model

@torch.inference_mode()
def chat(tokenizer, model, messages: list[dict], max_new_tokens=256, temperature=0.7):
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(model.device)

    input_len = inputs["input_ids"].shape[-1] 
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,     
    )

    gen_ids = out[0][input_len:]                
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text.strip()                         
