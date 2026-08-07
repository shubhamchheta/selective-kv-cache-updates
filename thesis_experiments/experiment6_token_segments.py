import os
import torch

from dotenv import load_dotenv
from transformers import AutoTokenizer

load_dotenv("../.env")

MODEL = "meta-llama/Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL,
    token=os.getenv("HF_TOKEN")
)

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

# --------------------------------------------------

token_ids = tokenizer.encode(knowledge)

print("="*70)
print("WHOLE DOCUMENT")
print("="*70)

print("Total Tokens:", len(token_ids))
print()

for i, t in enumerate(token_ids):
    print(f"{i:3d} : {t:6d} : {repr(tokenizer.decode([t]))}")

# --------------------------------------------------

segment_size = len(token_ids) // 3

seg1 = token_ids[:segment_size]
seg2 = token_ids[segment_size:segment_size*2]
seg3 = token_ids[segment_size*2:]

print("\n")
print("="*70)
print("SEGMENTS")
print("="*70)

for name, seg in zip(
    ["Segment1", "Segment2", "Segment3"],
    [seg1, seg2, seg3]
):

    print(name)
    print("Length:", len(seg))
    print("First:", seg[0], repr(tokenizer.decode([seg[0]])))
    print("Last :", seg[-1], repr(tokenizer.decode([seg[-1]])))
    print()
