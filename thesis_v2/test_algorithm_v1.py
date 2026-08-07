import copy

from tokenizer_utils import load_tokenizer

from dataset import (
    knowledge,
    segments
)

from cache_builder import CacheBuilder
from cache_mapper import CacheMapper

from cache_adapter import CacheAdapter

from cache_truncate import CacheTruncate
from cache_merge import CacheMerge
from tail_builder import TailBuilder

print("Loading tokenizer...")

tokenizer = load_tokenizer()

builder = CacheBuilder()
mapper = CacheMapper(tokenizer)

################################################

print("\nBuilding Whole Cache")

whole_ids = tokenizer.encode(
    knowledge,
    return_tensors="pt"
)

whole_cache = builder.build(whole_ids)

################################################

mapping = mapper.build_mapping()

changed = 1

keep = mapping[changed]["start"]

################################################

print("\nKeeping Prefix")

# Make a copy first
prefix_cache = copy.deepcopy(whole_cache)

# Truncate only the copy
prefix_cache = CacheTruncate.truncate(
    prefix_cache,
    keep
)

################################################

print("\nBuilding Tail")

tail = TailBuilder.build(
    segments,
    changed
)

tail_ids = tokenizer.encode(
    tail,
    return_tensors="pt"
)

tail_cache = builder.build(tail_ids)

################################################

print("\nMerging")

updated_cache = CacheMerge.merge(
    prefix_cache,
    tail_cache
)

################################################
print("\nVERIFY")

print("Whole :", CacheAdapter.get_seq_length(whole_cache))
print("Prefix:", CacheAdapter.get_seq_length(prefix_cache))
print("Tail  :", CacheAdapter.get_seq_length(tail_cache))
print("Final :", CacheAdapter.get_seq_length(updated_cache))
print()

for i in [0,15,31]:

    print("Layer",i)

    print(
        CacheAdapter.get_keys(
            updated_cache,
            i
        ).shape
    )
