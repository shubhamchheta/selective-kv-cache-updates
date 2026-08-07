import os
import torch

from dotenv import load_dotenv
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DynamicCache,
)

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
    device_map="auto",
    token=os.getenv("HF_TOKEN")
)

print("Model Loaded.")

device = model.model.embed_tokens.weight.device

#########################################################

chapter1 = """Chapter 1:
Cats are mammals.
Cats have four legs.
"""

chapter2 = """Chapter 2:
Dogs are mammals.
Dogs bark loudly.
Dogs like bones.
"""

chapter3 = """Chapter 3:
Birds can fly.
Birds lay eggs.
"""

knowledge = chapter1 + "\n" + chapter2 + "\n" + chapter3

#########################################################
# Whole cache
#########################################################

print("\nBuilding WHOLE cache...")

whole_ids = tokenizer.encode(
    knowledge,
    return_tensors="pt"
).to(device)

whole_cache = DynamicCache()

with torch.no_grad():

    outputs = model(
        input_ids=whole_ids,
        past_key_values=whole_cache,
        use_cache=True,
    )

whole_cache = outputs.past_key_values

print("Whole cache built.")

#########################################################
# Token-preserving split
#########################################################

whole_list = tokenizer.encode(knowledge)

bos = whole_list[:1]

content = whole_list[1:]

c1 = tokenizer.encode(chapter1, add_special_tokens=False)
c2 = tokenizer.encode(chapter2, add_special_tokens=False)
c3 = tokenizer.encode(chapter3, add_special_tokens=False)

offset = 0

seg1 = bos + content[offset:offset+len(c1)]
offset += len(c1)

seg2 = content[offset:offset+len(c2)]
offset += len(c2)

seg3 = content[offset:offset+len(c3)]

#########################################################
# Build cache for ONE segment
#########################################################

def build_cache(token_ids):

    ids = torch.tensor([token_ids]).to(device)

    cache = DynamicCache()

    with torch.no_grad():

        outputs = model(
            input_ids=ids,
            past_key_values=cache,
            use_cache=True,
        )

    return outputs.past_key_values

print("\nBuilding segment caches...")

cache1 = build_cache(seg1)
cache2 = build_cache(seg2)
cache3 = build_cache(seg3)

#########################################################
# Merge caches
#########################################################

merged = DynamicCache()

merged.key_cache = []
merged.value_cache = []

for layer in range(len(cache1.key_cache)):

    k = torch.cat([
        cache1.key_cache[layer],
        cache2.key_cache[layer],
        cache3.key_cache[layer],
    ], dim=2)

    v = torch.cat([
        cache1.value_cache[layer],
        cache2.value_cache[layer],
        cache3.value_cache[layer],
    ], dim=2)

    merged.key_cache.append(k)
    merged.value_cache.append(v)

merged._seen_tokens = (
    cache1._seen_tokens +
    cache2._seen_tokens +
    cache3._seen_tokens
)

#########################################################
# Compare
#########################################################

print()
print("="*70)
print("COMPARE")
print("="*70)

for layer in [0, 15, 31]:

    wk = whole_cache.key_cache[layer]
    mk = merged.key_cache[layer]

    print()
    print("Layer", layer)

    print("Whole :", wk.shape)
    print("Merge :", mk.shape)

    print("Equal Shape :", wk.shape == mk.shape)

    if wk.shape == mk.shape:

        print("Allclose :", torch.allclose(wk, mk))

        diff = (wk - mk).abs()

        print("Max diff :", diff.max().item())

        print("Mean diff:", diff.mean().item())
