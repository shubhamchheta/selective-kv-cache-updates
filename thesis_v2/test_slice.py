from tokenizer_utils import load_tokenizer
from dataset import knowledge

from cache_builder import CacheBuilder
from cache_mapper import CacheMapper
from cache_slice import CacheSlice

print("Loading tokenizer...")

tokenizer = load_tokenizer()

builder = CacheBuilder()
mapper = CacheMapper(tokenizer)

print()

print("=" * 60)
print("BUILD CACHE")
print("=" * 60)

ids = tokenizer.encode(
    knowledge,
    return_tensors="pt"
)

cache = builder.build(ids)

mapping = mapper.build_mapping()

chapter2 = mapping[1]

print()

print("=" * 60)
print("CHAPTER 2")
print("=" * 60)

print(chapter2)

print()

print("=" * 60)
print("EXTRACT SLICE")
print("=" * 60)

slice_cache = CacheSlice.extract(
    cache,
    chapter2["start"],
    chapter2["end"]
)

print("Layers :", len(slice_cache))
print("Slice Length :", CacheSlice.slice_length(slice_cache))

print()

for layer in [0, 15, 31]:

    print("Layer", layer)

    print("Keys")
    print(slice_cache[layer]["keys"].shape)

    print("Values")
    print(slice_cache[layer]["values"].shape)

    print("-" * 40)
