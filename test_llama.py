from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
import torch
import os

load_dotenv(".env")

model_name = "meta-llama/Llama-3.2-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    token=os.getenv("HF_TOKEN")
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    token=os.getenv("HF_TOKEN")
)

prompt = "What is the capital of Germany?"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

output = model.generate(
    **inputs,
    max_new_tokens=30
)

print(tokenizer.decode(output[0], skip_special_tokens=True))
