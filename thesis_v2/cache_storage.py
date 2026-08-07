import json
import torch


class CacheStorage:

    @staticmethod
    def save_cache(cache, path):

        torch.save(cache, path)

    @staticmethod
    def load_cache(path):

        return torch.load(path,weights_only=False)

    @staticmethod
    def save_index(index, path):

        with open(path, "w") as f:
            json.dump(index, f, indent=4)

    @staticmethod
    def load_index(path):

        with open(path) as f:
            return json.load(f)
