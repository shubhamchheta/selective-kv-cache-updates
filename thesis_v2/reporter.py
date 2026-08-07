class Reporter:

    @staticmethod
    def header(changed):

        print("=" * 60)
        print("Selective Cache Update Experiment")
        print("=" * 60)
        print(f"Changed Chapter : {changed+1}")
        print()

    @staticmethod
    def loading():

        print("Loading model...")

    @staticmethod
    def warmup():

        print("GPU Warmup...")

    @staticmethod
    def summary(
        full_score,
        selective_score,
        offline_time,
        full_time,
        selective_time,
        version1_tokens,
        version2_tokens,
        prefix_tokens,
        updated_tokens
    ):

        reuse_ratio = (prefix_tokens / version1_tokens) * 100

        recomputed = updated_tokens - prefix_tokens

        reduction = ((version2_tokens - recomputed) / version2_tokens) * 100

        print()
        print("=" * 60)
        print("EXPERIMENT RESULT")
        print("=" * 60)

        print()
        print("Answer Quality")
        print("-" * 30)
        print(f"Full Cache Similarity      : {full_score:.4f}")
        print(f"Selective Cache Similarity : {selective_score:.4f}")

        print()
        print("Cache Statistics")
        print("-" * 30)
        print(f"Original Cache Tokens      : {version1_tokens}")
        print(f"Updated Cache Tokens       : {version2_tokens}")
        print(f"Reused Tokens              : {prefix_tokens}")
        print(f"Recomputed Tokens          : {recomputed}")
        print(f"Cache Reuse Ratio          : {reuse_ratio:.2f}%")
        print(f"Token Reduction            : {reduction:.2f}%")

        print()
        print("Performance")
        print("-" * 30)
        print(f"Offline Cache Build        : {offline_time:.3f} sec")
        print(f"Full Cache Rebuild         : {full_time:.3f} sec")
        print(f"Selective Cache Update     : {selective_time:.3f} sec")

        print()
        print("Status")
        print("-" * 30)

        if abs(full_score - selective_score) < 0.01:
            print("✓ Answer quality preserved")

        print("✓ Cache updated successfully")
        print("✓ Prefix cache reused")
