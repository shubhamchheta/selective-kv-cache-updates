import torch

from transformers.cache_utils import DynamicCache

from segment_cache_manager import SegmentCacheManager
from cache_merger import CacheMerger

torch.serialization.add_safe_globals([DynamicCache])
torch.serialization.add_safe_globals([set])

manager = SegmentCacheManager()

print("Loading caches...")

chapter1 = manager.load("chapter1")
chapter2 = manager.load("chapter2")
chapter3 = manager.load("chapter3")

print()

print("="*70)
print("Original Lengths")
print("="*70)

print("Chapter1 :", chapter1.key_cache[0].shape[2])
print("Chapter2 :", chapter2.key_cache[0].shape[2])
print("Chapter3 :", chapter3.key_cache[0].shape[2])

merger = CacheMerger()

merged = merger.merge([
    chapter1,
    chapter2,
    chapter3
])

print()

print("="*70)
print("Merged Result")
print("="*70)

print("Layers :", len(merged.key_cache))

print()

print("Layer0")

print("Key :", merged.key_cache[0].shape)

print("Value :", merged.value_cache[0].shape)

print()

print("Last Layer")

print("Key :", merged.key_cache[-1].shape)

print("Value :", merged.value_cache[-1].shape)

torch.save(
    merged,
    "segment_cache/merged_cache.pt"
)

print()

print("Merged cache saved.")
