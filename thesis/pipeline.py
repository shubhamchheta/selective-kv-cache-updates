from segmentation import Segmenter
from cache_builder import CacheBuilder
from cache_storage import CacheStorage
from cache_merger import CacheMerger
from selective_update import SelectiveUpdater


class ThesisPipeline:

    def __init__(self):

        self.segmenter = Segmenter()

        self.builder = CacheBuilder()

        self.storage = CacheStorage()

        self.merger = CacheMerger()

        self.updater = SelectiveUpdater()

    ##################################################
 
    def build_cache(
        self,
        knowledge,
        segment_texts,
        names,
    ):

        print("\nStep 1 : Tokenize Whole Document")

        whole_ids = self.segmenter.tokenize_document(
            knowledge
        )

        print("Done")

        ##################################################

        print("\nStep 2 : Compute Segment Lengths")

        lengths = []

        for seg in segment_texts:

            lengths.append(
                len(
                    self.segmenter.tokenize_segment(seg)
                )
            )

        print(lengths)

        ##################################################

        print("\nStep 3 : Split Token IDs")

        segments = self.segmenter.split_by_lengths(
            whole_ids,
            lengths
        )

        print("Done")

        ##################################################

        print("\nStep 4 : Build KV Cache")

        caches = []

        for name, seg in zip(names, segments):

            print(f"Building {name}")

            cache = self.builder.build_cache(seg)

            caches.append(cache)

            self.storage.save(
                cache,
                name
            )

        ##################################################

        print("\nStep 5 : Merge")

        merged = self.merger.merge(
            caches
        )

        print("Done")

        return merged

        ##################################################

    def update_cache(

        self,

        old_segments,

        new_segments,

        names,

    ):

        print("\nStep 1 : Detect Changes")

        changed = self.updater.find_changed_segments(

            old_segments,

            new_segments

        )

        print("Changed:", changed)

        ##################################################

        print("\nStep 2 : Load Existing Caches")

        caches = []

        for name in names:

            caches.append(

                self.storage.load(name)

            )

        ##################################################

        print("\nStep 3 : Rebuild Changed Segments")

        rebuilt = [None] * len(caches)

        for idx in changed:

            print(f"Rebuilding {names[idx]}")

            whole = self.segmenter.tokenize_document(

                new_segments[idx]

            )

            cache = self.builder.build_cache(

                whole

            )

            rebuilt[idx] = cache

            self.storage.save(

                cache,

                names[idx]

            )

        ##################################################

        print("\nStep 4 : Replace Changed Cache")

        caches = self.updater.apply_updates(

            caches,

            rebuilt,

            changed

        )

        ##################################################

        print("\nStep 5 : Merge")

        merged = self.merger.merge(

            caches

        )

        print("Done")

        return merged
