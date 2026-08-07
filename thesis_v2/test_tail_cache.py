from tokenizer_utils import load_tokenizer
from cache_builder import CacheBuilder
from cache_adapter import CacheAdapter

from dataset import segments
from tail_builder import TailBuilder

print("Loading tokenizer...")

tokenizer = load_tokenizer()

builder = CacheBuilder()

tail = TailBuilder.build(
    segments,
    1
)

ids = tokenizer.encode(
    tail,
    return_tensors="pt"
)

cache = builder.build(ids)

print()

print("="*60)
print("TAIL CACHE")
print("="*60)

print("Layers:",
      CacheAdapter.num_layers(cache))

print("Seq:",
      CacheAdapter.get_seq_length(cache))

print()

for i in [0,15,31]:

    print("Layer",i)

    print(
        CacheAdapter.get_keys(
            cache,
            i
        ).shape
    )
