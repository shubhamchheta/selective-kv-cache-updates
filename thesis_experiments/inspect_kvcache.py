import os
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache

load_dotenv(".env")

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

print("=" * 80)
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    token=os.getenv("HF_TOKEN")
)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    token=os.getenv("HF_TOKEN")
)

print("=" * 80)

###########################################################################
# Small textbook
###########################################################################

knowledge = """
Chapter 1:
Cats are mammals.
Cats have four legs.

Chapter 2:
Dogs are mammals.
Dogs bark.

Chapter 3:
Birds can fly.
Birds lay eggs.
"""

print("\nKnowledge Text:\n")
print(knowledge)

###########################################################################
# Tokenization
###########################################################################

input_ids = tokenizer.encode(
    knowledge,
    return_tensors="pt"
).to(model.device)

tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

print("\n")
print("=" * 80)
print("TOKEN INFORMATION")
print("=" * 80)

print(f"Total Tokens: {len(tokens)}")
print()

for i, (tid, tok) in enumerate(zip(input_ids[0].tolist(), tokens)):
    print(f"{i:3d} | ID={tid:6d} | {tok}")

###########################################################################
# Build KV Cache
###########################################################################

print("\n")
print("=" * 80)
print("BUILDING KV CACHE")
print("=" * 80)

cache = DynamicCache()

with torch.no_grad():
    outputs = model(
        input_ids=input_ids,
        past_key_values=cache,
        use_cache=True
    )

kv = outputs.past_key_values

###########################################################################
# Inspect KV Cache
###########################################################################

print("\n")
print("=" * 80)
print("KV CACHE INFORMATION")
print("=" * 80)

print("Number of Transformer Layers:",
      len(kv.key_cache))

print()

for layer in range(len(kv.key_cache)):

    key = kv.key_cache[layer]
    value = kv.value_cache[layer]

    print(f"Layer {layer}")

    print(" Key Shape  :", key.shape)
    print(" Value Shape:", value.shape)

    print("-" * 60)

###########################################################################
# Check sequence length
###########################################################################

print("\n")
print("=" * 80)
print("SEQUENCE LENGTH CHECK")
print("=" * 80)

print("Token count          :", len(tokens))
print("KV sequence length   :", kv.key_cache[0].shape[-2])

print()

if len(tokens) == kv.key_cache[0].shape[-2]:
    print("✅ One KV position per input token.")
else:
    print("❌ Token count and KV length differ.")
