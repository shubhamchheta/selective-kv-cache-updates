from cache_adapter import CacheAdapter


class CacheSlice:
    """
    Extract KV slices from the whole cache.

    This class NEVER modifies the cache.
    It only returns the requested slice.
    """

    @staticmethod
    def extract(cache, start, end):

        slices = []

        for layer in range(CacheAdapter.num_layers(cache)):

            keys = CacheAdapter.get_keys(cache, layer)
            values = CacheAdapter.get_values(cache, layer)

            key_slice = keys[:, :, start:end + 1, :].clone()
            value_slice = values[:, :, start:end + 1, :].clone()

            slices.append(
                {
                    "keys": key_slice,
                    "values": value_slice
                }
            )

        return slices

    @staticmethod
    def slice_length(slice_cache):

        return slice_cache[0]["keys"].shape[2]
