import textbook_dataset

from cache_builder import CacheBuilder
from cache_mapper import CacheMapper
from cache_truncate import CacheTruncate
from tail_builder import TailBuilder
from cache_merge import CacheMerge

from change_detector import ChangeDetector

print("=" * 60)
print("LOAD DATASET")
print("=" * 60)

old_segments, _ = textbook_dataset.load("v1")
new_segments, _ = textbook_dataset.load("v2")

changed = ChangeDetector.detect(
    old_segments,
    new_segments
)

changed = changed[0]

print("Changed chapter:", changed + 1)

print()

##########################################################

knowledge_v1 = "\n\n\n\n".join(old_segments)

builder = CacheBuilder()

print("=" * 60)
print("BUILD WHOLE CACHE")
print("=" * 60)

whole_cache = builder.build_text(knowledge_v1)

##########################################################

mapper = CacheMapper()

mapping = mapper.map(
    knowledge_v1,
    old_segments
)

print()

print("=" * 60)
print("MAPPING")
print("=" * 60)

print(mapping)

##########################################################

keep_tokens = mapping[changed]["start"]

print()

print("=" * 60)
print("KEEP PREFIX")
print("=" * 60)

print("Tokens:", keep_tokens)

prefix_cache = CacheTruncate.truncate(
    whole_cache,
    keep_tokens
)

##########################################################

print()

print("=" * 60)
print("BUILD TAIL")
print("=" * 60)

tail_cache = TailBuilder(builder).build(
    new_segments,
    changed
)

##########################################################

print()

print("=" * 60)
print("MERGE")
print("=" * 60)

updated_cache = CacheMerge.merge(
    prefix_cache,
    tail_cache
)

print()

print("=" * 60)
print("DONE")
print("=" * 60)

print("Original:",
      mapping[-1]["end"] if -1 in mapping else "OK")

print("Updated cache built successfully.")
