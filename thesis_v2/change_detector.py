class ChangeDetector:

    @staticmethod
    def detect(old_segments, new_segments):

        changed = []

        for i in range(len(old_segments)):

            if old_segments[i] != new_segments[i]:

                changed.append(i)

        return changed
