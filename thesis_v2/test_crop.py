import sys

sys.path.append("..")
import textbook_dataset

from cache_builder import CacheBuilder
from cache_adapter import CacheAdapter
from cache_crop import CacheCrop

segments, _ = textbook_dataset.load("v1")

knowledge = "\n\n\n\n".join(segments)

builder = CacheBuilder()

cache = builder.build_text(knowledge)

print("Before")
print(CacheAdapter.get_keys(cache, 0).shape)

cache = CacheCrop.remove_prefix(
    cache,
    50
)

print("After")
print(CacheAdapter.get_keys(cache, 0).shape)
