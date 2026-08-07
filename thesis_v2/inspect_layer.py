from tokenizer_utils import load_tokenizer
from dataset import knowledge
from cache_builder import CacheBuilder

print("Loading tokenizer...")
tokenizer = load_tokenizer()

builder = CacheBuilder()

ids = tokenizer.encode(
    knowledge,
    return_tensors="pt"
)

cache = builder.build(ids)

print("\nCache type:")
print(type(cache))

print("\nNumber of layers:")
print(len(cache.layers))

layer = cache.layers[0]

print("\nLayer type:")
print(type(layer))

print("\nLayer attributes:")
print(dir(layer))

print("\nLayer object:")
print(layer)
