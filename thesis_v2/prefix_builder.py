class PrefixBuilder:
    """
    Build the prefix required for selective KV recomputation.

    Example

    changed = 1

    prefix =
        chapter1 +
        chapter2

    changed = 2

    prefix =
        chapter1 +
        chapter2 +
        chapter3
    """

    @staticmethod
    def build(segments, changed_segment):

        return "\n\n".join(
            segments[:changed_segment + 1]
        )
