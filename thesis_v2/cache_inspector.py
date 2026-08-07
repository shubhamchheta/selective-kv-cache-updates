class CacheInspector:

    def extract_slice(
        self,
        cache,
        start,
        end
    ):

        slices = []

        for layer in range(len(cache.key_cache)):

            key = cache.key_cache[layer][:, :, start:end+1, :]

            value = cache.value_cache[layer][:, :, start:end+1, :]

            slices.append((key, value))

        return slices
