import os
import time

from thesis_v2.cache_storage import CacheStorage
from thesis_v2.cache_cropper import CacheCropper
from thesis_v2.cache_adapter import CacheAdapter
from thesis_v2.build_index import IndexBuilder


class SelectiveUpdater:

    def __init__(self, builder):

        self.builder = builder

    # -------------------------------------------------
    # Build initial cache (Version 1)
    # -------------------------------------------------

    def build_initial_cache(self, chapters):

        t1 = time.time()

        cache = self.builder.build_text(
            "\n\n".join(chapters)
        )

        t2 = time.time()

        kv_time = t2 - t1

        index_builder = IndexBuilder(self.builder.tokenizer)

        index = index_builder.build(chapters)

        os.makedirs("cache", exist_ok=True)

        CacheStorage.save_cache(
            cache,
            "cache/textbook_cache.pt"
        )

        CacheStorage.save_index(
            index,
            "cache/textbook_index.json"
        )

        return cache, kv_time

    # -------------------------------------------------
    # Full rebuild (Version 2)
    # -------------------------------------------------

    def build_full_cache(self, chapters):

        t1 = time.time()

        cache = self.builder.build_text(
            "\n\n".join(chapters)
        )

        t2 = time.time()

        return cache, t2 - t1

    # -------------------------------------------------
    # Crop prefix cache
    # -------------------------------------------------

    def build_prefix_cache(self, changed):

        cache = CacheStorage.load_cache(
            "cache/textbook_cache.pt"
        )

        index = CacheStorage.load_index(
            "cache/textbook_index.json"
        )

        prefix_tokens = index[f"chapter_{changed}"]["end"]

        prefix_cache = CacheCropper.crop(
            cache,
            prefix_tokens
        )

        return prefix_cache, prefix_tokens

    # -------------------------------------------------
    # Selective Update
    # -------------------------------------------------

    def build_selective_cache(
        self,
        suffix,
        prefix_cache
    ):

        t1 = time.time()

        cache = self.builder.build_text(
            suffix,
            past_key_values=prefix_cache
        )

        t2 = time.time()

        return cache, t2 - t1
