import copy

from thesis_v2.cache_adapter import CacheAdapter


class CacheCropper:

    @staticmethod
    def crop(cache, token_count):

        cache = copy.deepcopy(cache)

        layers = CacheAdapter.num_layers(cache)

        for layer in range(layers):

            keys = CacheAdapter.get_keys(cache, layer)
            values = CacheAdapter.get_values(cache, layer)

            CacheAdapter.set_keys(
                cache,
                layer,
                keys[:, :, :token_count, :]
            )

            CacheAdapter.set_values(
                cache,
                layer,
                values[:, :, :token_count, :]
            )

        return cache
