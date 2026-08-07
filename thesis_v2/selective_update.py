import copy

from tokenizer_utils import load_tokenizer

from cache_builder import CacheBuilder
from cache_mapper import CacheMapper

from cache_truncate import CacheTruncate
from cache_merge import CacheMerge
from tail_builder import TailBuilder
from change_detector import ChangeDetector


class SelectiveUpdater:

    def __init__(self):

        self.tokenizer = load_tokenizer()

        self.builder = CacheBuilder()

        self.mapper = CacheMapper(
            self.tokenizer
        )

        self.document = None
        self.segments = None
        self.mapping = None
        self.cache = None

    ####################################################

    def build(
        self,
        document,
        segments
    ):

        self.document = document
        self.segments = segments

        ids = self.tokenizer.encode(
            document,
            return_tensors="pt"
        )

        self.cache = self.builder.build(ids)

        self.mapping = self.mapper.build_mapping()

        return self.cache

    ####################################################

    def update(
        self,
        new_document,
        new_segments
    ):

        changed = ChangeDetector.detect(
            self.segments,
            new_segments
        )

        # detect() returns a list of changed segments.
        # For now, support a single changed segment.
        changed = changed[0]

        print()
        print("Changed Segment :", changed)

        keep = self.mapping[changed]["start"]

        prefix = copy.deepcopy(
            self.cache
        )

        prefix = CacheTruncate.truncate(
            prefix,
            keep
        )

        tail = TailBuilder.build(
            new_segments,
            changed
        )

        tail_ids = self.tokenizer.encode(
            tail,
            return_tensors="pt"
        )

        tail_cache = self.builder.build(
            tail_ids
        )

        updated = CacheMerge.merge(
            prefix,
            tail_cache
        )

        self.document = new_document
        self.segments = new_segments
        self.cache = updated

        return updated
