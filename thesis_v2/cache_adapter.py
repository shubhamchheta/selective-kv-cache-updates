class CacheAdapter:
    """
    Compatibility layer for HuggingFace DynamicCache.

    All future code should use this class instead of directly
    accessing cache.layers.
    """

    @staticmethod
    def num_layers(cache):
        return len(cache.layers)

    @staticmethod
    def get_keys(cache, layer):
        return cache.layers[layer].keys

    @staticmethod
    def get_values(cache, layer):
        return cache.layers[layer].values

    @staticmethod
    def set_keys(cache, layer, keys):
        cache.layers[layer].keys = keys

    @staticmethod
    def set_values(cache, layer, values):
        cache.layers[layer].values = values

    @staticmethod
    def get_seq_length(cache):
        return cache.layers[0].keys.shape[2]
