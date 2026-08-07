import torch
from transformers.cache_utils import DynamicCache

torch.serialization.add_safe_globals([DynamicCache])
torch.serialization.add_safe_globals([set])

cache = torch.load(
    "segment_cache/merged_cache.pt",
    weights_only=True
)

print("=" * 60)
print("Chapter1 Cache")
print("=" * 60)

print(cache.__dict__)

print()

print("seen_tokens:", cache.seen_tokens)

print("_seen_tokens:", cache._seen_tokens)

print("seq_length:", cache.get_seq_length())

print()

print("Layer0 shape:", cache.key_cache[0].shape)
