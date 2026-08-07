import os
from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

load_dotenv(".env")

MODEL = "meta-llama/Llama-3.1-8B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL,
    token=os.getenv("HF_TOKEN")
)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    token=os.getenv("HF_TOKEN")
)

print("Generating...")

messages = [
    {"role": "user", "content": "What is the capital of Germany?"}
]

inputs = tokenizer.apply_chat_template(
    messages,
    return_tensors="pt",
    add_generation_prompt=True
).to(model.device)

outputs = model.generate(
    inputs,
    max_new_tokens=50,
    do_sample=False
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
