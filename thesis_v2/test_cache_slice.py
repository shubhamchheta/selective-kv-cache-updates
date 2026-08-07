from tokenizer_utils import load_tokenizer

from dataset import knowledge

from cache_builder import CacheBuilder

from cache_mapper import CacheMapper

from cache_inspector import CacheInspector

print("Loading tokenizer...")

tokenizer = load_tokenizer()

builder = CacheBuilder()

mapper = CacheMapper(tokenizer)

inspector = CacheInspector()

print()

print("="*60)
print("TOKENIZE")
print("="*60)

ids = tokenizer.encode(
    knowledge,
    return_tensors="pt"
)

print("Tokens :", ids.shape[1])

print()

print("="*60)
print("BUILD CACHE")
print("="*60)

cache = builder.build(ids)

print(type(cache))

print()

print(dir(cache))

print()

print(cache)

print()

mapping = mapper.build_mapping()

chapter2 = mapping[1]

print("="*60)
print("CHAPTER 2")
print("="*60)

print(chapter2)

print()

print("="*60)
print("EXTRACTING")
print("="*60)

slice_cache = inspector.extract_slice(
    cache,
    chapter2["start"],
    chapter2["end"]
)

for layer in [0, 15, 31]:

    key, value = slice_cache[layer]

    print()

    print("Layer", layer)

    print("Key  :", key.shape)

    print("Value:", value.shape)

print()

print("="*60)
print("DONE")
