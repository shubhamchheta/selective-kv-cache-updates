from pipeline import ThesisPipeline

pipeline = ThesisPipeline()

##########################################################

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

##########################################################

segments = [

    chapter1,

    chapter2,

    chapter3
]

names = [

    "chapter1",

    "chapter2",

    "chapter3"
]

##########################################################

merged = pipeline.build_cache(

    knowledge,

    segments,

    names

)

##########################################################

print()

print("="*60)

print("FINAL CACHE")

print("="*60)

print()

print(len(merged.key_cache))

print()

print(merged.key_cache[0].shape)

print()

print("Seen Tokens")

print(merged._seen_tokens)
