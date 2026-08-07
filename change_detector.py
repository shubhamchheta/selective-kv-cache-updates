class ChangeDetector:

    @staticmethod
    def first_changed(old_segments, new_segments):

        n = min(len(old_segments), len(new_segments))

        for i in range(n):

            if old_segments[i].strip() != new_segments[i].strip():
                return i

        if len(old_segments) != len(new_segments):
            return n

        return -1
