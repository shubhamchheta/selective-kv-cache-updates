import torch
from transformers.cache_utils import DynamicCache


class CacheMerger:

    def merge(self, caches):

        merged = DynamicCache()

        num_layers = len(caches[0].key_cache)

        print("=" * 70)
        print("MERGING KV CACHES")
        print("=" * 70)

        for layer in range(num_layers):

            merged_key = torch.cat(
                [cache.key_cache[layer] for cache in caches],
                dim=2
            )

            merged_value = torch.cat(
                [cache.value_cache[layer] for cache in caches],
                dim=2
            )

            merged.key_cache.append(merged_key)
            merged.value_cache.append(merged_value)

            print(
                f"Layer {layer:2d} : "
                f"{merged_key.shape}"
            )

        merged._seen_tokens = merged.key_cache[0].shape[2]
        return merged
