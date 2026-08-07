import os
import sys
import torch

# So we can import kvcache.py from the CAG root
sys.path.append("..")

from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM

from kvcache import generate

from segmentation import Segmenter
from cache_builder import CacheBuilder
from cache_merger import CacheMerger

load_dotenv("../.env")

MODEL = "meta-llama/Llama-3.1-8B-Instruct"

############################################################

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

############################################################

segmenter = Segmenter()
builder = CacheBuilder()
merger = CacheMerger()

############################################################

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

############################################################
# QUESTION
############################################################

question = """
Answer the question ONLY using the knowledge.

Question:
How many legs do cats have?

Answer:
"""

############################################################
# BUILD WHOLE CACHE
############################################################

print("\nBuilding Whole Cache...")

whole_ids = segmenter.tokenize_document(knowledge)

whole_cache = builder.build_cache(whole_ids)

############################################################
# BUILD MERGED CACHE
############################################################

print("Building Segmented Cache...")

lengths = [

    len(segmenter.tokenize_segment(chapter1)),
    len(segmenter.tokenize_segment(chapter2)),
    len(segmenter.tokenize_segment(chapter3))
]

segments = segmenter.split_by_lengths(
    whole_ids,
    lengths
)

caches = []

for seg in segments:

    caches.append(
        builder.build_cache(seg)
    )

merged_cache = merger.merge(caches)

############################################################

device = model.model.embed_tokens.weight.device

question_ids = tokenizer.encode(
    question,
    return_tensors="pt"
).to(device)

############################################################

print("\n" + "="*70)
print("QUESTION")
print("="*70)
print(question)

############################################################
# WHOLE CACHE
############################################################

print("\n" + "="*70)
print("WHOLE CACHE")
print("="*70)

answer = generate(
    model,
    question_ids,
    whole_cache,
    max_new_tokens=40
)

print(tokenizer.decode(answer[0], skip_special_tokens=True))

############################################################
# MERGED CACHE
############################################################

print("\n" + "="*70)
print("MERGED CACHE")
print("="*70)

answer = generate(
    model,
    question_ids,
    merged_cache,
    max_new_tokens=40
)

print(tokenizer.decode(answer[0], skip_special_tokens=True))
