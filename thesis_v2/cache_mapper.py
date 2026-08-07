from dataset import segments


class CacheMapper:

    def __init__(self, tokenizer):

        self.tokenizer = tokenizer

    def build_mapping(self):

        mapping = []

        position = 0

        for idx, segment in enumerate(segments):

            ids = self.tokenizer.encode(segment)

            # remove BOS except first segment
            if idx != 0:
                ids = ids[1:]

            start = position
            end = position + len(ids) - 1

            mapping.append({
                "segment": idx,
                "start": start,
                "end": end,
                "length": len(ids)
            })

            position += len(ids)

        return mapping
