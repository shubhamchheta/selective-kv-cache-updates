from dataset import segments
from tail_builder import TailBuilder

print("="*60)

for i in range(len(segments)):

    print()
    print("Changed:", i)
    print("-"*40)

    print(
        TailBuilder.build(
            segments,
            i
        )
    )
