import copy
import torch

from transformers.cache_utils import DynamicCache
from cache_adapter import CacheAdapter


class CacheMerge:

    @staticmethod
    def merge(prefix_cache, tail_cache):

        merged = DynamicCache()

        for layer in range(CacheAdapter.num_layers(prefix_cache)):

            pk = CacheAdapter.get_keys(prefix_cache, layer)
            pv = CacheAdapter.get_values(prefix_cache, layer)

            tk = CacheAdapter.get_keys(tail_cache, layer)
            tv = CacheAdapter.get_values(tail_cache, layer)

            keys = torch.cat((pk, tk), dim=2)
            values = torch.cat((pv, tv), dim=2)

            # IMPORTANT: create an independent copy
            new_layer = copy.deepcopy(prefix_cache.layers[layer])

            new_layer.keys = keys.clone()
            new_layer.values = values.clone()

            merged.layers.append(new_layer)

        return merged
