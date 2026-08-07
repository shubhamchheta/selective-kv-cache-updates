import torch

from tokenizer_utils import load_tokenizer
from dataset import knowledge

from cache_builder import CacheBuilder
from cache_mapper import CacheMapper

from cache_slice import CacheSlice
from cache_updater import CacheUpdater
from cache_adapter import CacheAdapter

print("Loading tokenizer...")

tokenizer = load_tokenizer()

builder = CacheBuilder()
mapper = CacheMapper(tokenizer)

ids = tokenizer.encode(
    knowledge,
    return_tensors="pt"
)

print()

print("="*60)
print("BUILD CACHE")
print("="*60)

cache = builder.build(ids)

mapping = mapper.build_mapping()

chapter2 = mapping[1]

print()

print("="*60)
print("EXTRACT")
print("="*60)

slice_cache = CacheSlice.extract(
    cache,
    chapter2["start"],
    chapter2["end"]
)

print("Done")

print()

print("="*60)
print("REPLACE")
print("="*60)

cache = CacheUpdater.replace(
    cache,
    chapter2["start"],
    chapter2["end"],
    slice_cache
)

print("Done")

print()

print("="*60)
print("VERIFY")
print("="*60)

for layer in [0,15,31]:

    whole = CacheAdapter.get_keys(cache,layer)

    restored = whole[:,:,chapter2["start"]:chapter2["end"]+1,:]

    equal = torch.allclose(
        restored,
        slice_cache[layer]["keys"]
    )

    print()

    print("Layer",layer)

    print("Shape :",restored.shape)

    print("Equal :",equal)
