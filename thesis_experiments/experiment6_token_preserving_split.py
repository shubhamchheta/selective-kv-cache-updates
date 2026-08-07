import os
from dotenv import load_dotenv
from transformers import AutoTokenizer

load_dotenv("../.env")

MODEL = "meta-llama/Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL,
    token=os.getenv("HF_TOKEN")
)

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

knowledge = (
    chapter1
    + "\n"
    + chapter2
    + "\n"
    + chapter3
)

print("="*70)
print("TOKENIZING WHOLE DOCUMENT")
print("="*70)

whole_ids = tokenizer.encode(knowledge)

print("Whole Tokens:", len(whole_ids))

####################################################
# Tokenize EACH chapter WITHOUT special tokens
####################################################

c1 = tokenizer.encode(
    chapter1,
    add_special_tokens=False
)

c2 = tokenizer.encode(
    chapter2,
    add_special_tokens=False
)

c3 = tokenizer.encode(
    chapter3,
    add_special_tokens=False
)

print()
print("="*70)
print("Chapter Token Lengths")
print("="*70)

print("Chapter1:", len(c1))
print("Chapter2:", len(c2))
print("Chapter3:", len(c3))

####################################################
# Whole document begins with BOS
####################################################

bos = whole_ids[:1]

content = whole_ids[1:]

####################################################
# Split using chapter lengths
####################################################

seg1 = bos + content[:len(c1)]

offset = len(c1)

seg2 = content[offset:offset+len(c2)]

offset += len(c2)

seg3 = content[offset:offset+len(c3)]

####################################################

print()
print("="*70)
print("Segments")
print("="*70)

for name, seg in zip(
    ["Segment1","Segment2","Segment3"],
    [seg1,seg2,seg3]
):

    print(name)
    print("Length :", len(seg))
    print("First  :", seg[0], repr(tokenizer.decode([seg[0]])))
    print("Last   :", seg[-1], repr(tokenizer.decode([seg[-1]])))
    print()

####################################################

merged = seg1 + seg2 + seg3

print("="*70)
print("Verification")
print("="*70)

print("Whole :", len(whole_ids))
print("Merged:", len(merged))

print()

print("Token sequence identical?")

print(merged == whole_ids)

####################################################
# Find first mismatch if any
####################################################

if merged != whole_ids:

    print()

    print("First mismatch:")

    for i, (a, b) in enumerate(zip(whole_ids, merged)):

        if a != b:

            print("Index:", i)
            print("Whole :", a, repr(tokenizer.decode([a])))
            print("Merge :", b, repr(tokenizer.decode([b])))
            break

else:

    print()
    print("SUCCESS")
