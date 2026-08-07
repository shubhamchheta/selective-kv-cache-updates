from pipeline import ThesisPipeline

pipeline = ThesisPipeline()

###################################################

old_segments = [

"""Chapter 1:
Cats are mammals.
Cats have four legs.
""",

"""Chapter 2:
Dogs are mammals.
Dogs bark loudly.
Dogs like bones.
""",

"""Chapter 3:
Birds can fly.
Birds lay eggs.
"""
]

###################################################

new_segments = [

old_segments[0],

"""Chapter 2:
Dogs are mammals.
Dogs bark loudly.
Dogs are friendly.
Dogs like bones.
""",

old_segments[2]

]

###################################################

names = [

"chapter1",

"chapter2",

"chapter3"

]

###################################################

knowledge = "\n".join(old_segments)

pipeline.build_cache(

    knowledge,

    old_segments,

    names

)

###################################################

print()

print("="*60)

print("UPDATING CACHE")

print("="*60)

updated = pipeline.update_cache(

    old_segments,

    new_segments,

    names

)

###################################################

print()

print("="*60)

print("UPDATED CACHE")

print("="*60)

print()

print(updated.key_cache[0].shape)

print(updated._seen_tokens)
