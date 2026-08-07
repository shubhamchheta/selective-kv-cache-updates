import os
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

############################################################

token_ids = tokenizer.encode(knowledge)

print("="*70)
print("Whole Token Count:", len(token_ids))
print("="*70)

############################################################
# Find Chapter 2
############################################################

chapter2_text = "Chapter 2:"
chapter3_text = "Chapter 3:"

chapter2_ids = tokenizer.encode(
    chapter2_text,
    add_special_tokens=False
)

chapter3_ids = tokenizer.encode(
    chapter3_text,
    add_special_tokens=False
)

print("Chapter2 IDs:", chapter2_ids)
print("Decoded:")
for t in chapter2_ids:
    print(t, repr(tokenizer.decode([t])))

print()

print("Chapter3 IDs:", chapter3_ids)
print("Decoded:")
for t in chapter3_ids:
    print(t, repr(tokenizer.decode([t])))

def find_sublist(big, small):

    for i in range(len(big)-len(small)+1):

        if big[i:i+len(small)] == small:
            return i

    return -1

chapter2_start = find_sublist(token_ids, chapter2_ids)
chapter3_start = find_sublist(token_ids, chapter3_ids)

print("Chapter2 starts at :", chapter2_start)
print("Chapter3 starts at :", chapter3_start)

############################################################
# Split exactly at chapter boundaries
############################################################

segment1 = token_ids[:chapter2_start]
segment2 = token_ids[chapter2_start:chapter3_start]
segment3 = token_ids[chapter3_start:]

############################################################

segments = [
    ("Segment1", segment1),
    ("Segment2", segment2),
    ("Segment3", segment3)
]

print()
print("="*70)
print("SEGMENTS")
print("="*70)

for name, seg in segments:

    print(name)

    print("Length :", len(seg))

    print("First :", seg[0], repr(tokenizer.decode([seg[0]])))

    print("Last  :", seg[-1], repr(tokenizer.decode([seg[-1]])))

    print()

############################################################

merged = segment1 + segment2 + segment3

print("="*70)
print("Verification")
print("="*70)

print("Merged Length :", len(merged))
print("Whole Length  :", len(token_ids))

print()

print("Exactly Equal ?")

print(merged == token_ids)
