from cache_adapter import CacheAdapter


class CacheUpdater:
    """
    Replace a KV slice inside the whole cache.

    This class assumes the replacement slice has
    exactly the same length.
    """

    @staticmethod
    def replace(
        cache,
        start,
        end,
        slice_cache
    ):

        for layer in range(CacheAdapter.num_layers(cache)):

            keys = CacheAdapter.get_keys(cache, layer)
            values = CacheAdapter.get_values(cache, layer)

            keys[:, :, start:end+1, :] = slice_cache[layer]["keys"]

            values[:, :, start:end+1, :] = slice_cache[layer]["values"]

            CacheAdapter.set_keys(cache, layer, keys)
            CacheAdapter.set_values(cache, layer, values)

        return cache
