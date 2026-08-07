from segmentation import Segmenter

segmenter = Segmenter()

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

whole = segmenter.tokenize_document(
    knowledge
)

lengths = [

    len(segmenter.tokenize_segment(chapter1)),

    len(segmenter.tokenize_segment(chapter2)),

    len(segmenter.tokenize_segment(chapter3))
]

segments = segmenter.split_by_lengths(
    whole,
    lengths
)

merged = []

for seg in segments:

    merged.extend(seg)

print("="*60)

print("Whole:", len(whole))

print("Merged:", len(merged))

print()

print("Identical:")

print(whole == merged)
