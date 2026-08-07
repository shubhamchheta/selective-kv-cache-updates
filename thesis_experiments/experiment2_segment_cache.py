import os
import torch

from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

load_dotenv("../.env")

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

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

print("\nModel loaded.\n")


chapters = {
    "Chapter1": """
Cats are mammals.
Cats have four legs.
""",

    "Chapter2": """
Dogs are mammals.
Dogs bark.
""",

    "Chapter3": """
Birds can fly.
Birds lay eggs.
"""
}


segment_caches = {}

for name, text in chapters.items():

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

    segment_caches[name] = cache

    seq_len = cache.key_cache[0].shape[2]

    print("Tokens :", inputs["input_ids"].shape[1])
    print("KV Len :", seq_len)

    print()

    for layer in [0, 15, 31]:

        print(f"Layer {layer}")

        print(
            "Key   ",
            cache.key_cache[layer].shape
        )

        print(
            "Value ",
            cache.value_cache[layer].shape
        )

        print()

print("="*70)
print("Finished.")
