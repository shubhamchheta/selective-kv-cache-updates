import os
import torch

from dotenv import load_dotenv
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)
from transformers.cache_utils import DynamicCache

torch.serialization.add_safe_globals([DynamicCache])
torch.serialization.add_safe_globals([set])

load_dotenv("../.env")

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
    device_map="cuda",
    token=os.getenv("HF_TOKEN")
)

print("Model Loaded.\n")

# ------------------------------------------------------
# SAME knowledge used in previous experiments
# ------------------------------------------------------

knowledge = """
Chapter 1:
Cats are mammals.
Cats have four legs.

Chapter 2:
Dogs are mammals.
Dogs bark loudly.
Dogs like bones.

Chapter 3:
Birds can fly.
Birds lay eggs.
"""

# ------------------------------------------------------

print("Building WHOLE cache...")

inputs = tokenizer(
    knowledge,
    return_tensors="pt"
).to(model.device)

whole_cache = DynamicCache()

with torch.no_grad():

    outputs = model(
        **inputs,
        past_key_values=whole_cache,
        use_cache=True
    )

whole_cache = outputs.past_key_values

print("Whole cache built.\n")

print("Loading merged cache...")

merged_cache = torch.load(
    "segment_cache/merged_cache.pt",
    weights_only=True
)

print()

print("="*70)
print("COMPARISON")
print("="*70)

for layer in [0, 15, 31]:

    wk = whole_cache.key_cache[layer]
    mk = merged_cache.key_cache[layer]

    wv = whole_cache.value_cache[layer]
    mv = merged_cache.value_cache[layer]

    print(f"\nLayer {layer}")

    print("Key shape")
    print(wk.shape, mk.shape)

    print("Value shape")
    print(wv.shape, mv.shape)

    print()

    print("Key allclose :",
          torch.allclose(wk, mk))

    print("Value allclose :",
          torch.allclose(wv, mv))

    print()

    print("Max key difference :",
          torch.abs(wk - mk).max().item())

    print("Max value difference :",
          torch.abs(wv - mv).max().item())
