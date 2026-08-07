import textbook_dataset

from thesis_v2.change_detector import ChangeDetector

old_segments, _ = textbook_dataset.load("v1")

new_segments, _ = textbook_dataset.load("v2")

changed = ChangeDetector.detect(
    old_segments,
    new_segments
)

print()

print("Changed Segments")

print(changed)
