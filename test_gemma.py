import os
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM

load_dotenv(".env")

model_name = "google/gemma-2-9b-it"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    token=os.getenv("HF_TOKEN")
)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=os.getenv("HF_TOKEN"),
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

messages = [
    {
        "role": "user",
        "content": "What is the capital of Germany?"
    }
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

outputs = model.generate(
    inputs,
    max_new_tokens=50,
    do_sample=False
)

answer = tokenizer.decode(
    outputs[0][inputs.shape[-1]:],
    skip_special_tokens=True
)

print("\n========== ANSWER ==========")
print(answer)
