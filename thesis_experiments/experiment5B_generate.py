import os
import torch

from dotenv import load_dotenv
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)
from transformers.cache_utils import DynamicCache

load_dotenv("../.env")

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

torch.serialization.add_safe_globals([DynamicCache])
torch.serialization.add_safe_globals([set])

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    token=os.getenv("HF_TOKEN")
)

print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="cuda",
    token=os.getenv("HF_TOKEN")
)

print("\nLoading merged cache...")

merged_cache = torch.load(
    "segment_cache/merged_cache.pt",
    weights_only=True
)

print("Merged cache loaded.")

print()

print("Sequence Length:",
      merged_cache.key_cache[0].shape[2])

question = "What do dogs do?"

prompt = (
    question
    + "<|eot_id|>"
    + "<|start_header_id|>assistant<|end_header_id|>\n"
)

inputs = tokenizer(
    prompt,
    return_tensors="pt"
).to(model.device)

print()

print("Generating...")

with torch.no_grad():

    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        past_key_values=merged_cache,
        max_new_tokens=50,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

generated = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[1]:],
    skip_special_tokens=True
)

print()

print("="*70)

print("QUESTION")

print(question)

print()

print("ANSWER")

print(generated)

print("="*70)
