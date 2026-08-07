from change_detector import ChangeDetector

old = [

"""Chapter 1:
Cats are mammals.
Cats have four legs.""",

"""Chapter 2:
Dogs are mammals.
Dogs bark.""",

"""Chapter 3:
Birds can fly.
Birds lay eggs."""
]

new = [

"""Chapter 1:
Cats are mammals.
Cats have four legs.""",

"""Chapter 2:
Dogs are mammals.
Dogs walk.""",

"""Chapter 3:
Birds can fly.
Birds lay eggs."""
]

changed = ChangeDetector.detect(
    old,
    new
)

print()

print("="*60)
print("CHANGE DETECTOR")
print("="*60)

print(changed)
