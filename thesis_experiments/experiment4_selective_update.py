import os
import torch

from dotenv import load_dotenv
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)
from transformers.cache_utils import DynamicCache

from segment_cache_manager import SegmentCacheManager

load_dotenv("../.env")

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    token=os.getenv("HF_TOKEN")
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="cuda",
    token=os.getenv("HF_TOKEN")
)

manager = SegmentCacheManager()

print("\nCurrent Cache Files")
manager.list_segments()

print("\n")

#######################################################
# Updated Chapter 2
#######################################################

updated_text = """
Dogs are mammals.

Dogs bark loudly.

Dogs are loyal animals.

Dogs are commonly kept as pets.
"""

print("="*70)
print("Recomputing ONLY Chapter 2")
print("="*70)

inputs = tokenizer(
    updated_text,
    return_tensors="pt"
).to(model.device)

cache = DynamicCache()

with torch.no_grad():

    outputs = model(
        **inputs,
        past_key_values=cache,
        use_cache=True
    )

cache = outputs.past_key_values

manager.save("chapter2", cache)

print("\n")

print("="*70)
print("Cache After Update")
print("="*70)

manager.list_segments()

print("\n")

manager.info("chapter2")
