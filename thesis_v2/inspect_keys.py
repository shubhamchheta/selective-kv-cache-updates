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

print()

print("="*60)
print("CACHE INFORMATION")
print("="*60)

print("Layers :", len(cache.layers))

print()

for layer_id in [0, 15, 31]:

    layer = cache.layers[layer_id]

    print(f"Layer {layer_id}")

    print("Keys  :", layer.keys.shape)
    print("Values:", layer.values.shape)

    print("-"*40)
