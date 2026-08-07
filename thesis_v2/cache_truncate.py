from cache_adapter import CacheAdapter


class CacheTruncate:

    @staticmethod
    def truncate(cache, keep_tokens):

        for layer in range(CacheAdapter.num_layers(cache)):

            keys = CacheAdapter.get_keys(cache, layer)
            values = CacheAdapter.get_values(cache, layer)

            CacheAdapter.set_keys(
                cache,
                layer,
                keys[:, :, :keep_tokens, :].clone()
            )

            CacheAdapter.set_values(
                cache,
                layer,
                values[:, :, :keep_tokens, :].clone()
            )

        return cache
