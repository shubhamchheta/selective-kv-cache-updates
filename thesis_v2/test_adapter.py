from tokenizer_utils import load_tokenizer
from dataset import knowledge

from cache_builder import CacheBuilder
from cache_adapter import CacheAdapter

print("Loading tokenizer...")
tokenizer = load_tokenizer()

builder = CacheBuilder()

ids = tokenizer.encode(
    knowledge,
    return_tensors="pt"
)

cache = builder.build(ids)

print()

print("=" * 60)
print("CACHE ADAPTER TEST")
print("=" * 60)

print("Layers :", CacheAdapter.num_layers(cache))
print("Sequence Length :", CacheAdapter.get_seq_length(cache))

print()

for layer in [0, 15, 31]:

    print(f"Layer {layer}")

    keys = CacheAdapter.get_keys(cache, layer)
    values = CacheAdapter.get_values(cache, layer)

    print("Keys  :", keys.shape)
    print("Values:", values.shape)

    print("-" * 40)
