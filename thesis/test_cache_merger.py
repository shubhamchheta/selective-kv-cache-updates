from segmentation import Segmenter
from cache_builder import CacheBuilder
from cache_merger import CacheMerger

segmenter = Segmenter()
builder = CacheBuilder()
merger = CacheMerger()

####################################################

chapter1 = """Chapter 1:
Cats are mammals.
Cats have four legs.
"""

chapter2 = """Chapter 2:
Dogs are mammals.
Dogs bark loudly.
Dogs like bones.
"""

chapter3 = """Chapter 3:
Birds can fly.
Birds lay eggs.
"""

knowledge = chapter1 + "\n" + chapter2 + "\n" + chapter3

####################################################

whole = segmenter.tokenize_document(knowledge)

lengths = [
    len(segmenter.tokenize_segment(chapter1)),
    len(segmenter.tokenize_segment(chapter2)),
    len(segmenter.tokenize_segment(chapter3)),
]

segments = segmenter.split_by_lengths(
    whole,
    lengths
)

####################################################

caches = []

for seg in segments:

    caches.append(
        builder.build_cache(seg)
    )

####################################################

merged = merger.merge(caches)

####################################################

print()

print("=" * 60)

print("Merged Cache")

print("=" * 60)

print()

print("Layers :", len(merged.key_cache))

print()

print("Layer0")

print(merged.key_cache[0].shape)

print(merged.value_cache[0].shape)

print()

print("Last Layer")

print(merged.key_cache[-1].shape)

print(merged.value_cache[-1].shape)

print()

print("Seen Tokens")

print(merged._seen_tokens)
