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
changed = len(v1) - 7
v1[changed] = v1[changed] + """

Virtual memory allows programs to use more memory than is physically
available. It uses disk storage to extend the available memory...

"""
v2[changed] = v2[changed] + """

Virtual memory allows programs to use more memory than is physically
available. It uses secondary storage to extend the available memory...

"""

print("=" * 60)
print("SQuAD SELECTIVE CACHE EXPERIMENT")
print("=" * 60)

print("Total Chapters :", len(v1))
print("Changed Chapter:", changed + 1)

print()
print("Version 1 tokens will represent the original knowledge base.")
print("Version 2 contains a modification to the chapter.")

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



torch.cuda.synchronize()

t7 = time.time()

cache_v2 = builder.build_text("\n\n".join(v2))

torch.cuda.synchronize()

t8 = time.time()

times.append(t8 - t7)



print()
print("Average:", sum(times)/len(times))
print("Minimum:", min(times))
print("Maximum:", max(times))




#t8 = time.time()

#full_rebuild_time = t8 - t7

#print("Time   :", round(t8 - t7, 3), "sec")
print("Tokens :", CacheAdapter.get_keys(cache_v2, 0).shape[2])

# ============================================================
# DEBUG: KV CACHE STRUCTURE
# ============================================================

print("\n" + "=" * 60)
print("KV CACHE STRUCTURE")
print("=" * 60)

v1_keys = CacheAdapter.get_keys(cache_v1, 0)
v2_keys = CacheAdapter.get_keys(cache_v2, 0)

print("V1 Key shape :", v1_keys.shape)
print("V2 Key shape :", v2_keys.shape)

print("Cache V1 type:", type(cache_v1))
print("Cache V2 type:", type(cache_v2))

# ============================================================
# COMPARE V1 vs V2 KV CACHE
# ============================================================

print()
print("=" * 60)
print("KV CACHE DIFFERENCE ANALYSIS")
print("=" * 60)

# ------------------------------------------------------------
# Get keys
# ------------------------------------------------------------

v1_keys = CacheAdapter.get_keys(cache_v1, 0)
v2_keys = CacheAdapter.get_keys(cache_v2, 0)

print("V1 key shape:", v1_keys.shape)
print("V2 key shape:", v2_keys.shape)

# ------------------------------------------------------------
# Compare token by token
# ------------------------------------------------------------

token_differences = []

for token_idx in range(v1_keys.shape[2]):

    diff = torch.max(
        torch.abs(
            v1_keys[:, :, token_idx, :] -
            v2_keys[:, :, token_idx, :]
        )
    ).item()

    token_differences.append(diff)

# ------------------------------------------------------------
# Count changed tokens
# ------------------------------------------------------------

threshold = 1e-6

changed_tokens = [
    i for i, diff in enumerate(token_differences)
    if diff > threshold
]

unchanged_tokens = [
    i for i, diff in enumerate(token_differences)
    if diff <= threshold
]

print()
print("Total tokens      :", len(token_differences))
print("Changed KV tokens :", len(changed_tokens))
print("Same KV tokens    :", len(unchanged_tokens))

print()
print("First changed token :", changed_tokens[0] if changed_tokens else "None")
print("Last changed token  :", changed_tokens[-1] if changed_tokens else "None")

print()
print("KV Change Ratio     :",
      len(changed_tokens) / len(token_differences) * 100,
      "%")

print()
print("=" * 60)
print("CHAPTER TOKEN BOUNDARIES")
print("=" * 60)

chapter_ranges = []

for i in range(len(v1)):

    # Start of chapter
    if i == 0:
        start = 0
    else:
        start = index[f"chapter_{i}"]["end"]

    # End of chapter
    end = index[f"chapter_{i + 1}"]["end"]

    chapter_ranges.append((start, end))

    print(
        f"Chapter {i + 1}: "
        f"tokens {start} -> {end}"
    )


# ============================================================
# KV DIFFERENCE BY CHAPTER
# ============================================================

print()
print("=" * 60)
print("KV DIFFERENCE BY CHAPTER")
print("=" * 60)

k1 = CacheAdapter.get_keys(cache_v1, 0)
k2 = CacheAdapter.get_keys(cache_v2, 0)

# Absolute KV difference
diff = torch.abs(k1 - k2)

# Reduce:
# [batch, heads, tokens, head_dim]
# ->
# one value per token

token_diff = diff.mean(dim=(0, 1, 3))


threshold = 1e-6

for i, (start, end) in enumerate(chapter_ranges):

    chapter_diff = token_diff[start:end]

    changedd = (chapter_diff > threshold).sum().item()

    avg_diff = chapter_diff.mean().item()

    max_diff = chapter_diff.max().item()

    print(
        f"Chapter {i + 1}: "
        f"Changed tokens = {changedd}/{end-start}, "
        f"Avg diff = {avg_diff:.8f}, "
        f"Max diff = {max_diff:.8f}"
    )

# ============================================================
# KV DIFFERENCE ACROSS TRANSFORMER LAYERS
# ============================================================

print()
print("=" * 60)
print("KV DIFFERENCE ACROSS TRANSFORMER LAYERS")
print("=" * 60)

layers_to_check = [0, 8, 16, 24, 31]

threshold = 1e-6

for layer in layers_to_check:

    k1 = CacheAdapter.get_keys(cache_v1, layer)
    k2 = CacheAdapter.get_keys(cache_v2, layer)

    # Absolute difference
    diff = torch.abs(k1 - k2)

    # [batch, heads, tokens, head_dim]
    # -> one value per token
    token_diff = diff.mean(dim=(0, 1, 3))

    changed_tokens = (
        token_diff > threshold
    ).sum().item()

    changed_positions = torch.where(
        token_diff > threshold
    )[0]

    print()
    print(f"Layer {layer}")
    print("-" * 40)

    print(
        "Changed tokens :",
        changed_tokens,
        "/",
        token_diff.shape[0]
    )

    if changed_tokens > 0:

        print(
            "First changed :",
            changed_positions[0].item()
        )

        print(
            "Last changed  :",
            changed_positions[-1].item()
        )

    print(
        "Change ratio   :",
        changed_tokens / token_diff.shape[0] * 100,
        "%"
    )


# ============================================================
# KV DIFFERENCE BY CHAPTER AND LAYER
# ============================================================

print("\n" + "=" * 60)
print("KV DIFFERENCE BY CHAPTER AND LAYER")
print("=" * 60)

# Chapter token boundaries
chapters_ranges = [
    (0, 420),       # Chapter 1
    (420, 962),     # Chapter 2
    (962, 1255),    # Chapter 3
    (1255, 1743),   # Chapter 4
    (1743, 2197),   # Chapter 5
    (2197, 2758),   # Chapter 6
    (2758, 3126),   # Chapter 7
    (3126, 3573),   # Chapter 8
]

# Layers we want to inspect
layers_to_check = [0, 8, 16, 24, 31]

k1_all = CacheAdapter.get_keys(cache_v1, 0)
k2_all = CacheAdapter.get_keys(cache_v2, 0)

for layer in layers_to_check:

    print("\n" + "-" * 60)
    print(f"LAYER {layer}")
    print("-" * 60)

    # [batch, heads, tokens, head_dim]
    k1 = CacheAdapter.get_keys(cache_v1, layer)
    k2 = CacheAdapter.get_keys(cache_v2, layer)

    # Difference for every token
    diff = torch.abs(k1 - k2)

    # Average over batch, heads and head dimension
    token_diff = diff.mean(dim=(0, 1, 3))

    for chapter_idx, (start, end) in enumerate(chapters_ranges):

        chapter_diff = token_diff[start:end]

        changedd = (chapter_diff > 1e-6).sum().item()

        avg_diff = chapter_diff.mean().item()
        max_diff = chapter_diff.max().item()

        print(
            f"Chapter {chapter_idx + 1}: "
            f"Changed = {changedd}/{end-start}, "
            f"Avg = {avg_diff:.8f}, "
            f"Max = {max_diff:.8f}"
        )

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

print("\nEXPERIMENT : TOKENS WITHOUT CACHE")

t111 = time.time()

builder.build_text(suffix)

torch.cuda.synchronize()

t112 = time.time()

print("Time :", round(t112 - t111, 6), "sec")


print("\nEXPERIMENT : TOKENS WITH CACHE")

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


cache = CacheStorage.load_cache("cache/textbook_cache.pt")

prefix_cache = CacheCropper.crop(cache,prefix_tokens)
torch.cuda.synchronize()
    
t11 = time.time()
print("Prefix cache tokens:",CacheAdapter.get_keys(prefix_cache, 0).shape[2])
updated_cache = builder.build_text(suffix,past_key_values=prefix_cache)
torch.cuda.synchronize()
t12 = time.time()
times.append(t12 - t11)
   

print()
print("Average:", sum(times)/len(times))
print("Minimum:", min(times))
print("Maximum:", max(times))
#print("Time   :", round(t12 - t11, 3), "sec")
print("Tokens :", CacheAdapter.get_keys(updated_cache, 0).shape[2])

tail_tokens=CacheAdapter.get_keys(updated_cache, 0).shape[2] - prefix_tokens

print(f"Prefix Tokens     : {prefix_tokens}")
print(f"Tail Tokens       : {tail_tokens}")

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


print()

full_tokens = prefix_tokens+tail_tokens

reuse = ( prefix_tokens * 100) / full_tokens 
print(f"Cache Reuse Ratio : {reuse:.2f}%")

reduction = (1 - (tail_tokens / full_tokens)) * 100
print(f"Token Reduction : {reduction:.2f}%")

#throughput = full_tokens / full_rebuild_time
#print(f"Throughput : {throughput:.2f} tokens/sec")

print()

print(f"Offline Build Time      : {offline_build_time:.3f} sec")
#print(f"Full Rebuild Time       : {full_rebuild_time:.3f} sec")
#print(f"Selective Update Time   : {selective_update_time:.3f} sec")
