from pipeline import ThesisPipeline
from dataset import knowledge

print()

print("="*60)
print("PIPELINE")
print("="*60)

pipeline = ThesisPipeline()

cache, mapping = pipeline.build(knowledge)

print()

print("Layers:", len(cache.layers))

print("Sequence:", cache.layers[0].keys.shape[2])

print()

chapter2 = pipeline.extract_segment(
    cache,
    mapping,
    1
)

print("Extracted")

cache = pipeline.replace_segment(
    cache,
    mapping,
    1,
    chapter2
)

print("Replacement Success")

print()

print("Finished.")
