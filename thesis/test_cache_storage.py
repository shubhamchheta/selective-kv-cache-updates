from segmentation import Segmenter
from cache_builder import CacheBuilder
from cache_storage import CacheStorage

segmenter = Segmenter()
builder = CacheBuilder()
storage = CacheStorage()

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
    len(segmenter.tokenize_segment(chapter3)),
]

segments = segmenter.split_by_lengths(
    whole,
    lengths
)

names = [
    "chapter1",
    "chapter2",
    "chapter3"
]

####################################################

for name, seg in zip(names, segments):

    cache = builder.build_cache(seg)

    storage.save(cache, name)

####################################################

print()

storage.list()

####################################################

print()

cache = storage.load("chapter2")

print()

print("Layers :", len(cache.key_cache))

print(cache.key_cache[0].shape)

####################################################

print()

print("Exists chapter2 :", storage.exists("chapter2"))

print("Exists chapter10:", storage.exists("chapter10"))
