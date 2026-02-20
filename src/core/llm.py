import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_llm(model_name: str, lora_path: str | None = None):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
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
def chat(tokenizer, model, prompt: str, max_new_tokens=256, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
    )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text[len(prompt):].strip()
