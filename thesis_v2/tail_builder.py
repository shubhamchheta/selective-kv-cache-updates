class TailBuilder:
    """
    Build the tail starting from the changed segment.
    """

    @staticmethod
    def build(segments, changed_segment):

        return "\n\n".join(
            segments[changed_segment:]
        )
