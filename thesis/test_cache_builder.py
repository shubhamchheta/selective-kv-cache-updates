from segmentation import Segmenter
from cache_builder import CacheBuilder

segmenter = Segmenter()

builder = CacheBuilder()

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

whole = segmenter.tokenize_document(knowledge)

lengths = [

    len(segmenter.tokenize_segment(chapter1)),
    len(segmenter.tokenize_segment(chapter2)),
    len(segmenter.tokenize_segment(chapter3))
]

segments = segmenter.split_by_lengths(
    whole,
    lengths
)

print()

for i, seg in enumerate(segments):

    print("=" * 60)
    print(f"Building Cache {i+1}")
    print("=" * 60)

    cache = builder.build_cache(seg)

    print("Layers :", len(cache.key_cache))

    print("Layer0")

    print(cache.key_cache[0].shape)

    print(cache.value_cache[0].shape)

    print()

    print("Last Layer")

    print(cache.key_cache[-1].shape)

    print(cache.value_cache[-1].shape)

    print()
