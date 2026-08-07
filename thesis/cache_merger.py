import torch
from transformers import DynamicCache


class CacheMerger:

    def merge(self, caches):

        """
        Merge any number of DynamicCache objects.

        Args:
            caches : list[DynamicCache]

        Returns:
            DynamicCache
        """

        merged = DynamicCache()

        merged.key_cache = []
        merged.value_cache = []

        num_layers = len(caches[0].key_cache)

        ################################################

        for layer in range(num_layers):

            keys = []

            values = []

            for cache in caches:

                keys.append(cache.key_cache[layer])

                values.append(cache.value_cache[layer])

            merged.key_cache.append(
                torch.cat(keys, dim=2)
            )

            merged.value_cache.append(
                torch.cat(values, dim=2)
            )

        ################################################

        merged._seen_tokens = sum(
            cache._seen_tokens
            for cache in caches
        )

        return merged
