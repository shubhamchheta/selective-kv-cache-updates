import os
import torch
from transformers.cache_utils import DynamicCache

# Allow loading DynamicCache objects
torch.serialization.add_safe_globals([DynamicCache])
torch.serialization.add_safe_globals([set])


class SegmentCacheManager:

    def __init__(self, cache_dir="segment_cache"):

        self.cache_dir = cache_dir

        os.makedirs(cache_dir, exist_ok=True)

    #####################################################

    def _path(self, segment_name):

        return os.path.join(
            self.cache_dir,
            f"{segment_name}.pt"
        )

    #####################################################

    def save(self, segment_name, cache):

        path = self._path(segment_name)

        torch.save(cache, path)

        print(f"Saved : {path}")

    #####################################################

    def load(self, segment_name):

        path = self._path(segment_name)

        if not os.path.exists(path):
            raise FileNotFoundError(path)

        cache = torch.load(
            path,
            weights_only=True
        )

        print(f"Loaded : {path}")

        return cache

    #####################################################

    def delete(self, segment_name):

        path = self._path(segment_name)

        if os.path.exists(path):

            os.remove(path)

            print(f"Deleted : {path}")

        else:

            print("File not found.")

    #####################################################

    def update(self, segment_name, cache):

        self.save(segment_name, cache)

        print("Updated.")

    #####################################################

    def list_segments(self):

        files = sorted(os.listdir(self.cache_dir))

        print("\nSegments")

        print("--------------------")

        for f in files:

            print(f)

        return files

    #####################################################

    def info(self, segment_name):

        cache = self.load(segment_name)

        print()

        print(segment_name)

        print("--------------------")

        print("Layers :", len(cache.key_cache))

        print("Layer0 Key :", cache.key_cache[0].shape)

        print("Layer0 Value:", cache.value_cache[0].shape)

        print("Last Layer Key :", cache.key_cache[-1].shape)

        print("Last Layer Value:", cache.value_cache[-1].shape)
