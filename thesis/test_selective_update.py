from selective_update import SelectiveUpdater

updater = SelectiveUpdater()

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

"""Chapter 1:
Cats are mammals.
Cats have four legs.
""",

"""Chapter 2:
Dogs are mammals.
Dogs bark loudly.
Dogs are friendly.
Dogs like bones.
""",

"""Chapter 3:
Birds can fly.
Birds lay eggs.
"""
]

###################################################

changed = updater.find_changed_segments(
    old_segments,
    new_segments
)

print()

print("="*60)

print("Changed Segments")

print("="*60)

print()

print(changed)
