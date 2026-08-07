import cag.similarity as cagsim
import time
import os
import logging
from transformers import logging as hf_logging


import torch
from transformers import AutoTokenizer
from selective_kvcache import generate

import textbook_dataset
from change_detector import ChangeDetector

from thesis_v2.cache_builder import CacheBuilder
from thesis_v2.cache_adapter import CacheAdapter

from thesis_v2.build_index import IndexBuilder
from thesis_v2.cache_storage import CacheStorage
from thesis_v2.cache_cropper import CacheCropper
from thesis_v2.squad_loader import SquadLoader

######################
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

hf_logging.set_verbosity_error()
#####################


full_scores = []
selective_scores = []

# ============================================================
# Load textbook datasets
# ============================================================
loader = SquadLoader()
chapters, questions = loader.load(max_questions=10)
chapters = chapters[:8]
v1 = chapters
v2 = v1.copy()

#v1, questions = textbook_dataset.load("v1")
#v2, _ = textbook_dataset.load("v2")


# ============================================================
# Detect first changed chapter
# ============================================================

#changed = ChangeDetector.first_changed(v1, v2)
changed = len(v1) - 3
v2[changed] = v2[changed] + """

This paragraph was added during the document update.
The purpose of this modification is to evaluate selective KV-cache regeneration.
"""

print("=" * 60)
print("SQuAD SELECTIVE CACHE EXPERIMENT")
print("=" * 60)

print("Total Chapters :", len(v1))
print("Changed Chapter:", changed + 1)

print()
print("Version 1 tokens will represent the original knowledge base.")
print("Version 2 contains a modification to the final chapter.")

#print("=" * 60)
#print("Changed Chapter :", changed + 1)
#print("=" * 60)

print()
print("Chapter sizes:")
print("-" * 40)

for i, chapter in enumerate(v1):

    print(
        f"Chapter {i + 1:02d} : "
        f"{len(chapter.split())} words"
    )
# ============================================================
# Create Builder
# ============================================================

builder = CacheBuilder()
tokenizer = builder.tokenizer

model = builder.model

print("GPU Warmup...")

builder.build_text("Warmup")

torch.cuda.synchronize()

print("Warmup Finished")

# ============================================================
# Build FULL cache for Version 1
# ============================================================

print("\nBUILDING FULL CACHE (VERSION 1)")

t1 = time.time()

cache_v1 = builder.build_text(
    "\n\n".join(v1)
)

t2 = time.time()

kv_build_time = t2 - t1

os.makedirs("cache", exist_ok=True)

t3 = time.time()

index_builder = IndexBuilder(builder.tokenizer)

index = index_builder.build(v1)

t4 = time.time()

index_time = t4 - t3


# ------------------------------
# Save Cache
# ------------------------------

t5 = time.time()

CacheStorage.save_cache(
    cache_v1,
    "cache/textbook_cache.pt"
)

CacheStorage.save_index(
    index,
    "cache/textbook_index.json"
)

t6 = time.time()

save_time = t6 - t5

offline_build_time = kv_build_time + index_time + save_time


print("\nCache Saved")

print(f"KV Build Time      : {kv_build_time:.3f} sec")
print(f"Index Build Time   : {index_time:.3f} sec")
print(f"Cache Save Time    : {save_time:.3f} sec")
print(f"Offline Total Time : {offline_build_time:.3f} sec")
print("Tokens :", CacheAdapter.get_keys(cache_v1, 0).shape[2])


# ============================================================
# Build FULL cache for Version 2
# (Baseline CAG after textbook modification)
# ============================================================

print("\nBUILDING FULL CACHE (VERSION 2)")

#t7 = time.time()

#cache_v2 = builder.build_text(
#    "\n\n".join(v2)
#)


#experiment mate 
times = []

for i in range(10):

    torch.cuda.synchronize()

    t7 = time.time()

    cache_v2 = builder.build_text(
        "\n\n".join(v2)
    )

    torch.cuda.synchronize()

    t8 = time.time()

    times.append(t8 - t7)

    print(f"Run {i+1}: {t8-t7:.6f} sec")

print()
print("Average:", sum(times)/len(times))
print("Minimum:", min(times))
print("Maximum:", max(times))




#t8 = time.time()

#full_rebuild_time = t8 - t7

#print("Time   :", round(t8 - t7, 3), "sec")
print("Tokens :", CacheAdapter.get_keys(cache_v2, 0).shape[2])




# ============================================================
# Split textbook
#
# Prefix  = unchanged chapters
# Suffix  = changed chapter + remaining chapters
# ============================================================

suffix = "\n\n".join(v2[changed:])
prefix = "\n\n".join(v1[:changed])


print()
print("Prefix Chapters :", changed)
print("Suffix Chapters :", len(v2) - changed)


# ============================================================
# Build Prefix Cache
#
# This cache will be reused.
# ============================================================

print("\nBUILDING PREFIX CACHE")


cache = CacheStorage.load_cache("cache/textbook_cache.pt")

index = CacheStorage.load_index("cache/textbook_index.json")

t9 = time.time()

prefix_tokens = index[f"chapter_{changed}"]["end"]

prefix_cache = CacheCropper.crop(
    cache,
    prefix_tokens
)

t10 = time.time()

print("Time   :", round(t10 - t9, 3), "sec")
print("Tokens :", CacheAdapter.get_keys(prefix_cache, 0).shape[2])







# ============================================================
# EXPERIMENT
# 21 TOKENS WITHOUT CACHE vs WITH CACHE
# ============================================================

print("\nEXPERIMENT : 21 TOKENS WITHOUT CACHE")

t111 = time.time()

builder.build_text(suffix)

torch.cuda.synchronize()

t112 = time.time()

print("Time :", round(t112 - t111, 6), "sec")


print("\nEXPERIMENT : 21 TOKENS WITH CACHE")

t113 = time.time()

builder.build_text(
    suffix,
    past_key_values=prefix_cache
)

torch.cuda.synchronize()

t114 = time.time()

print("Time :", round(t114 - t113, 6), "sec")






# ============================================================
# Selective Update
#
# Continue KV cache from prefix.
# Only changed chapter and remaining chapters
# are recomputed.
# ============================================================

print("\nBUILDING UPDATED TAIL")

times = []

for i in range(10):
    # Always use a fresh prefix cache
    cache = CacheStorage.load_cache("cache/textbook_cache.pt")

    prefix_cache = CacheCropper.crop(
        cache,
        prefix_tokens
    )
    torch.cuda.synchronize()
    
    t11 = time.time()
    print("Prefix cache tokens:",CacheAdapter.get_keys(prefix_cache, 0).shape[2])
    updated_cache = builder.build_text(
        suffix,
        past_key_values=prefix_cache
    )

    torch.cuda.synchronize()
    t12 = time.time()
    times.append(t12 - t11)
    print(f"Run {i+1}: {t12-t11:.6f} sec")

print()
print("Average:", sum(times)/len(times))
print("Minimum:", min(times))
print("Maximum:", max(times))
#print("Time   :", round(t12 - t11, 3), "sec")
print("Tokens :", CacheAdapter.get_keys(updated_cache, 0).shape[2])


print(f"Prefix Tokens     : {prefix_tokens}")
print(f"Tail Tokens       : {CacheAdapter.get_keys(updated_cache, 0).shape[2] - prefix_tokens}")

# ============================================================
# Summary
# ============================================================

print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)

print("Version 1 Cache :", CacheAdapter.get_keys(cache_v1, 0).shape[2])
print("Version 2 Cache :", CacheAdapter.get_keys(cache_v2, 0).shape[2])
print("Selective Cache :", CacheAdapter.get_keys(updated_cache, 0).shape[2])




print("\n" + "=" * 60)
print("TEST GENERATION")
print("=" * 60)

question = questions[0][0]


print("\n" + "="*60)
print("COMPARE ALL QUESTIONS")
print("="*60)

for i, (question, gt) in enumerate(questions):

    prompt  = f"""
<|start_header_id|>system<|end_header_id|>

You are a question answering assistant.

Answer the question ONLY using the information stored in the cached textbook.

Rules:
1. Do NOT use your own knowledge.
2. Do NOT guess.
3. If the answer is not explicitly present in the textbook, reply exactly:
I don't know.
4. Copy names, numbers and terminology exactly as they appear in the textbook.
5. Keep the answer short.

<|eot_id|>

<|start_header_id|>user<|end_header_id|>

Question:
{question}

<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
"""

    input_ids = tokenizer(
        prompt,
        return_tensors="pt"
    ).input_ids

    # ---------------- Full ----------------

    output = generate(
        model,
        input_ids,
        cache_v2,
        max_new_tokens=50
    )

    full_answer = tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )

    full_score = cagsim.bert(
        full_answer,
        gt
    )
    full_scores.append(full_score)
    # ---------------- Selective ----------------

    output = generate(
        model,
        input_ids,
        updated_cache,
        max_new_tokens=50
    )

    selective_answer = tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )
    selective_score = cagsim.bert(
        selective_answer,
        gt
    )
    selective_scores.append(selective_score)

    print("\n--------------------------------------------------")
    print("Q", i+1)
    print(question)

    print("\nFULL")
    print(full_answer)

    print("\nSELECTIVE")
    print(selective_answer)
    
    print("\nFULL SCORE :", round(full_score,4))
    print("SELECTIVE SCORE :", round(selective_score,4))


print("\n" + "=" * 60)
print("FINAL RESULT")
print("=" * 60)

print("Average Full Score      :", sum(full_scores) / len(full_scores))
print("Average Selective Score :", sum(selective_scores) / len(selective_scores))

print()

reuse = prefix_tokens / CacheAdapter.get_keys(cache_v1,0).shape[2] * 100
print(f"Cache Reuse Ratio : {reuse:.2f}%")

print()

full_tokens = CacheAdapter.get_keys(cache_v2,0).shape[2]
tail_tokens = (CacheAdapter.get_keys(updated_cache, 0).shape[2]- prefix_tokens)

reduction = (1 - (tail_tokens / full_tokens)) * 100
print(f"Token Reduction : {reduction:.2f}%")

print()

#throughput = full_tokens / full_rebuild_time
#print(f"Throughput : {throughput:.2f} tokens/sec")

print()

print(f"Offline Build Time      : {offline_build_time:.3f} sec")
#print(f"Full Rebuild Time       : {full_rebuild_time:.3f} sec")
#print(f"Selective Update Time   : {selective_update_time:.3f} sec")
