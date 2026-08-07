from tokenizer_utils import load_tokenizer
from cache_mapper import CacheMapper

print("Loading tokenizer...")

tokenizer = load_tokenizer()

mapper = CacheMapper(tokenizer)

mapping = mapper.build_mapping()

print()

print("="*60)
print("SEGMENT -> KV POSITION MAP")
print("="*60)

for m in mapping:

    print()

    print("Segment :", m["segment"] + 1)
    print("Start   :", m["start"])
    print("End     :", m["end"])
    print("Length  :", m["length"])

print()

total = mapping[-1]["end"] + 1

print("="*60)
print("Total Tokens :", total)
