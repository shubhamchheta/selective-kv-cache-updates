from selective_update import SelectiveUpdater

from dataset import (
    knowledge,
    segments
)

########################################################

new_segments = [

"""
Chapter 1:
Cats are mammals.
Cats have four legs.
""",

"""
Chapter 2:
Dogs are mammals.
Dogs have TWO legs.
Dogs like bones.
""",

"""
Chapter 3:
Birds can fly.
Birds lay eggs.
"""
]

new_document = "\n\n".join(new_segments)

########################################################

updater = SelectiveUpdater()

print("="*60)
print("INITIAL BUILD")
print("="*60)

cache = updater.build(
    knowledge,
    segments
)

print()

print("Sequence:",
      cache.layers[0].keys.shape[2])

########################################################

print()

print("="*60)
print("UPDATE")
print("="*60)

cache = updater.update(
    new_document,
    new_segments
)

print()

print("Sequence:",
      cache.layers[0].keys.shape[2])
