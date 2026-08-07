from segment_cache_manager import SegmentCacheManager

manager = SegmentCacheManager()

print("="*70)
print("Listing Segments")
print("="*70)

manager.list_segments()

print()

print("="*70)
print("Information")
print("="*70)

manager.info("chapter1")

print()

manager.info("chapter2")

print()

manager.info("chapter3")

print()

print("="*70)
print("Delete Test")
print("="*70)

manager.delete("chapter2")

print()

manager.list_segments()

print()

print("="*70)
print("Finished")
print("="*70)
