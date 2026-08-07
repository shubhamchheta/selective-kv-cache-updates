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

########################################################
# Load Model
########################################################

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

device = model.model.embed_tokens.weight.device

print("Model Loaded.")

########################################################
# Manual Generate (same idea as kvcache.py)
########################################################

def generate(cache, question, max_new_tokens=50):

    input_ids = tokenizer.encode(
        question,
        return_tensors="pt"
    ).to(device)

    next_token = input_ids
    output = input_ids.clone()

    with torch.no_grad():

        for _ in range(max_new_tokens):

            outputs = model(
                input_ids=next_token,
                past_key_values=cache,
                use_cache=True,
            )

            logits = outputs.logits[:, -1, :]

            next_token = logits.argmax(-1).unsqueeze(-1)

            cache = outputs.past_key_values

            output = torch.cat([output, next_token], dim=1)

            if next_token.item() in model.config.eos_token_id:
                break

    return tokenizer.decode(
        output[0][input_ids.shape[1]:],
        skip_special_tokens=True
    )

########################################################
# Knowledge
########################################################

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

########################################################
# Whole Cache
########################################################

print("\nBuilding Whole Cache...")

ids = tokenizer.encode(
    knowledge,
    return_tensors="pt"
).to(device)

whole_cache = DynamicCache()

with torch.no_grad():

    outputs = model(
        input_ids=ids,
        past_key_values=whole_cache,
        use_cache=True
    )

whole_cache = outputs.past_key_values

########################################################
# Segment Cache
########################################################

whole_ids = tokenizer.encode(knowledge)

bos = whole_ids[:1]
content = whole_ids[1:]

c1 = tokenizer.encode(chapter1, add_special_tokens=False)
c2 = tokenizer.encode(chapter2, add_special_tokens=False)
c3 = tokenizer.encode(chapter3, add_special_tokens=False)

offset = 0

seg1 = bos + content[offset:offset+len(c1)]
offset += len(c1)

seg2 = content[offset:offset+len(c2)]
offset += len(c2)

seg3 = content[offset:offset+len(c3)]

def build_cache(token_ids):

    ids = torch.tensor([token_ids]).to(device)

    cache = DynamicCache()

    with torch.no_grad():

        outputs = model(
            input_ids=ids,
            past_key_values=cache,
            use_cache=True
        )

    return outputs.past_key_values

print("Building Segment Caches...")

cache1 = build_cache(seg1)
cache2 = build_cache(seg2)
cache3 = build_cache(seg3)

########################################################
# Merge
########################################################

merged = DynamicCache()

merged.key_cache = []
merged.value_cache = []

for i in range(len(cache1.key_cache)):

    merged.key_cache.append(torch.cat([
        cache1.key_cache[i],
        cache2.key_cache[i],
        cache3.key_cache[i]
    ], dim=2))

    merged.value_cache.append(torch.cat([
        cache1.value_cache[i],
        cache2.value_cache[i],
        cache3.value_cache[i]
    ], dim=2))

merged._seen_tokens = (
    cache1._seen_tokens +
    cache2._seen_tokens +
    cache3._seen_tokens
)

########################################################
# Ask Question
########################################################

question = "How many legs do cats have?"

print()
print("="*70)
print("QUESTION")
print("="*70)

print(question)

print()
print("="*70)
print("WHOLE CACHE")
print("="*70)

answer1 = generate(whole_cache, question)

print(answer1)

print()
print("="*70)
print("MERGED CACHE")
print("="*70)

answer2 = generate(merged, question)

print(answer2)
