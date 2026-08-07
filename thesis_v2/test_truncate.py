from tokenizer_utils import load_tokenizer

from dataset import knowledge

from cache_builder import CacheBuilder
from cache_mapper import CacheMapper

from cache_adapter import CacheAdapter
from cache_truncate import CacheTruncate

print("Loading tokenizer...")

tokenizer = load_tokenizer()

builder = CacheBuilder()
mapper = CacheMapper(tokenizer)

ids = tokenizer.encode(
    knowledge,
    return_tensors="pt"
)

cache = builder.build(ids)

mapping = mapper.build_mapping()

changed = 1

keep = mapping[changed]["start"]

print()
print("="*60)
print("BEFORE")
print("="*60)

print(
    CacheAdapter.get_seq_length(cache)
)

cache = CacheTruncate.truncate(
    cache,
    keep
)

print()

print("="*60)
print("AFTER")
print("="*60)

print(
    CacheAdapter.get_seq_length(cache)
)

print()

for i in [0,15,31]:

    print("Layer",i)

    print(
        CacheAdapter.get_keys(
            cache,
            i
        ).shape
    )
