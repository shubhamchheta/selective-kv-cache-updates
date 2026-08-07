import os
import torch


class CacheStorage:

    def __init__(self, folder="segment_cache"):

        self.folder = folder

        os.makedirs(folder, exist_ok=True)

    ##################################################

    def save(self, cache, name):

        path = os.path.join(self.folder, f"{name}.pt")

        torch.save(cache, path)

        print(f"Saved : {path}")

    ##################################################

    def load(self, name):

        path = os.path.join(self.folder, f"{name}.pt")

        cache = torch.load(path,weights_only=False)

        print(f"Loaded : {path}")

        return cache

    ##################################################

    def list(self):

        print()

        print("Stored Caches")

        print("-"*40)

        for f in sorted(os.listdir(self.folder)):
            print(f)

    ##################################################

    def delete(self, name):

        path = os.path.join(self.folder, f"{name}.pt")

        if os.path.exists(path):

            os.remove(path)

            print(f"Deleted : {path}")

    ##################################################

    def exists(self, name):

        path = os.path.join(self.folder, f"{name}.pt")

        return os.path.exists(path)
