from cache_adapter import CacheAdapter


class CacheCrop:

    @staticmethod
    def remove_prefix(cache, prefix_tokens):

        for layer in range(CacheAdapter.num_layers(cache)):

            keys = CacheAdapter.get_keys(cache, layer)
            values = CacheAdapter.get_values(cache, layer)

            CacheAdapter.set_keys(
                cache,
                layer,
                keys[:, :, prefix_tokens:, :].clone()
            )

            CacheAdapter.set_values(
                cache,
                layer,
                values[:, :, prefix_tokens:, :].clone()
            )

        return cache
