from dataset import segments
from prefix_builder import PrefixBuilder

print("=" * 60)
print("PREFIX BUILDER")
print("=" * 60)

for changed in [0, 1, 2]:

    print()
    print("Changed Segment:", changed)
    print("-" * 40)

    prefix = PrefixBuilder.build(
        segments,
        changed
    )

    print(prefix)
