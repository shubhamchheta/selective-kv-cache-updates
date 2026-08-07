class SelectiveUpdater:

    ##################################################

    def find_changed_segments(
        self,
        old_segments,
        new_segments,
    ):

        changed = []

        for i, (old, new) in enumerate(
            zip(old_segments, new_segments)
        ):

            if old != new:
                changed.append(i)

        return changed

    ##################################################

    def apply_updates(
        self,
        caches,
        new_caches,
        changed_indices,
    ):
        """
        Replace only the changed caches.
        """

        updated = list(caches)

        for idx in changed_indices:
            updated[idx] = new_caches[idx]

        return updated
