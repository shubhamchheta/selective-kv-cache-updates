import torch

from tokenizer_utils import load_tokenizer

from cache_builder import CacheBuilder
from cache_adapter import CacheAdapter
from cache_mapper import CacheMapper

from prefix_builder import PrefixBuilder

from dataset import (
    knowledge,
    segments
)

print("Loading tokenizer...")
tokenizer = load_tokenizer()

builder = CacheBuilder()
mapper = CacheMapper(tokenizer)

########################################################
# BUILD WHOLE CACHE
########################################################

print()
print("="*60)
print("WHOLE CACHE")
print("="*60)

whole_ids = tokenizer.encode(
    knowledge,
    return_tensors="pt"
)

whole_cache = builder.build(whole_ids)

mapping = mapper.build_mapping()

########################################################
# BUILD PREFIX CACHE
########################################################

print()
print("="*60)
print("PREFIX CACHE")
print("="*60)

changed_segment = 1

prefix_text = PrefixBuilder.build(
    segments,
    changed_segment
)

prefix_ids = tokenizer.encode(
    prefix_text,
    return_tensors="pt"
)

prefix_cache = builder.build(prefix_ids)

########################################################
# INFORMATION
########################################################

whole_info = mapping[changed_segment]

chapter_length = whole_info["length"]

whole_start = whole_info["start"]
whole_end = whole_info["end"]

prefix_total = CacheAdapter.get_seq_length(prefix_cache)

prefix_start = prefix_total - chapter_length
prefix_end = prefix_total - 1

print()
print("="*60)
print("POSITIONS")
print("="*60)

print("Whole")
print(whole_start, whole_end)

print()

print("Prefix")
print(prefix_start, prefix_end)

########################################################
# COMPARE
########################################################

print()
print("="*60)
print("COMPARE")
print("="*60)

for layer in [0,15,31]:

    print()
    print("Layer",layer)

    ####################################################

    whole_keys = CacheAdapter.get_keys(
        whole_cache,
        layer
    )

    prefix_keys = CacheAdapter.get_keys(
        prefix_cache,
        layer
    )

    ####################################################

    whole_slice = whole_keys[
        :,
        :,
        whole_start:whole_end+1,
        :
    ]

    prefix_slice = prefix_keys[
        :,
        :,
        prefix_start:prefix_end+1,
        :
    ]

    ####################################################

    print("Whole :",whole_slice.shape)
    print("Prefix:",prefix_slice.shape)

    equal = torch.allclose(
        whole_slice,
        prefix_slice
    )

    diff = (
        whole_slice-prefix_slice
    ).abs()

    print("Allclose :",equal)
    print("Max Diff :",diff.max().item())
    print("Mean Diff:",diff.mean().item())
