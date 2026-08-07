import json
import os


class IndexBuilder:

    def __init__(self, tokenizer):

        self.tokenizer = tokenizer

    def build(self, chapters):

        index = {}

        start = 0

        for i, chapter in enumerate(chapters):

            length = self.tokenizer(
                chapter,
                return_tensors="pt",
                add_special_tokens=False
            ).input_ids.shape[1]

            index[f"chapter_{i+1}"] = {
                "start": start,
                "end": start + length,
                "length": length
            }

            start += length

        return index
