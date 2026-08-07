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

whole = chapter1 + "\n" + chapter2 + "\n" + chapter3

print("=" * 70)
print("WHOLE DOCUMENT")
print("=" * 70)

whole_ids = tokenizer.encode(whole)

print("Token Count:", len(whole_ids))
print()

print("=" * 70)
print("CHAPTERS")
print("=" * 70)

chapter_ids = []

for name, text in [
    ("Chapter1", chapter1),
    ("Chapter2", chapter2),
    ("Chapter3", chapter3),
]:
    ids = tokenizer.encode(text)
    if name != "Chapter1":
        ids = ids[1:]
    chapter_ids.extend(ids)

    print(name)
    print("Count:", len(ids))
    print("First Token:", ids[0], tokenizer.decode([ids[0]]))
    print("Last Token :", ids[-1], tokenizer.decode([ids[-1]]))
    print()

print("=" * 70)

print("Merged Count :", len(chapter_ids))
print("Whole Count  :", len(whole_ids))

print("=" * 70)

print("Whole IDs")
print(whole_ids)

print()

print("Merged IDs")
print(chapter_ids)
