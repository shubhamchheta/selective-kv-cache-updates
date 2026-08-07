from tokenizer_utils import load_tokenizer

from cache_builder import CacheBuilder
from cache_mapper import CacheMapper

from cache_slice import CacheSlice
from cache_updater import CacheUpdater


class ThesisPipeline:

    def __init__(self):

        self.tokenizer = load_tokenizer()

        self.builder = CacheBuilder()

        self.mapper = CacheMapper(self.tokenizer)

    def build(self, knowledge):

        ids = self.tokenizer.encode(
            knowledge,
            return_tensors="pt"
        )

        cache = self.builder.build(ids)

        mapping = self.mapper.build_mapping()

        return cache, mapping

    def extract_segment(
        self,
        cache,
        mapping,
        segment
    ):

        info = mapping[segment]

        return CacheSlice.extract(
            cache,
            info["start"],
            info["end"]
        )

    def replace_segment(
        self,
        cache,
        mapping,
        segment,
        new_slice
    ):

        info = mapping[segment]

        return CacheUpdater.replace(
            cache,
            info["start"],
            info["end"],
            new_slice
        )
