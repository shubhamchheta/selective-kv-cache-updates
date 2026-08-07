import torch

from selective_update import SelectiveUpdater
from cache_builder import CacheBuilder
from tokenizer_utils import load_tokenizer
from cache_adapter import CacheAdapter

from dataset import knowledge, segments

####################################################

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

####################################################

print("Loading...")

tokenizer = load_tokenizer()

builder = CacheBuilder()

####################################################
# Full rebuild
####################################################

ids = tokenizer.encode(
    new_document,
    return_tensors="pt"
)

full_cache = builder.build(ids)

####################################################
# Selective update
####################################################

updater = SelectiveUpdater()

updater.build(
    knowledge,
    segments
)

updated_cache = updater.update(
    new_document,
    new_segments
)

####################################################
# Compare
####################################################

print()
print("="*60)
print("COMPARE")
print("="*60)

for layer in [0,15,31]:

    fk = CacheAdapter.get_keys(full_cache, layer)
    uk = CacheAdapter.get_keys(updated_cache, layer)

    print()
    print("Layer", layer)

    print("Full :", fk.shape)
    print("Update:", uk.shape)

    if fk.shape == uk.shape:

        diff = (fk-uk).abs()

        print("Allclose :", torch.allclose(fk,uk))
        print("Max Diff :", diff.max().item())
        print("Mean Diff:", diff.mean().item())

    else:

        print("Shape mismatch")
