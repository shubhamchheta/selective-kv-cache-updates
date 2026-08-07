import os
import torch

from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache

load_dotenv("../.env")

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

SAVE_DIR = "segment_cache"

os.makedirs(SAVE_DIR, exist_ok=True)

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

print("\nModel Loaded\n")

segments = {
    "chapter1":
"""
Cats are mammals.
Cats have four legs.
""",

    "chapter2":
"""
Dogs are mammals.
Dogs bark.
""",

    "chapter3":
"""
Birds can fly.
Birds lay eggs.
"""
}

torch.serialization.add_safe_globals([DynamicCache])
torch.serialization.add_safe_globals([set])

for name, text in segments.items():

    print("="*70)
    print(name)
    print("="*70)

    inputs = tokenizer(
        text,
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

    save_path = os.path.join(
        SAVE_DIR,
        f"{name}.pt"
    )

    torch.save(cache, save_path)

    print("Saved:", save_path)

print("\n")
print("="*70)
print("Loading Saved Caches")
print("="*70)

for name in segments:

    path = os.path.join(
        SAVE_DIR,
        f"{name}.pt"
    )

    loaded_cache = torch.load(
        path,
        weights_only=True
    )

    print(name)

    print(
        loaded_cache.key_cache[0].shape
    )

    print(
        loaded_cache.value_cache[0].shape
    )

    print()
