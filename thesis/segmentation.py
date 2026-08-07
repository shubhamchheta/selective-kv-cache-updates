import os

from dotenv import load_dotenv
from transformers import AutoTokenizer

load_dotenv("../.env")

MODEL = "meta-llama/Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL,
    token=os.getenv("HF_TOKEN")
)


class Segmenter:

    def __init__(self):
        self.tokenizer = tokenizer

    def tokenize_document(self, text):

        return self.tokenizer.encode(text)

    def tokenize_segment(self, text):

        return self.tokenizer.encode(
            text,
            add_special_tokens=False
        )

    def split_by_lengths(self, whole_ids, segment_lengths):

        bos = whole_ids[:1]

        content = whole_ids[1:]

        segments = []

        offset = 0

        for i, length in enumerate(segment_lengths):

            if i == 0:

                seg = bos + content[offset:offset+length]

            else:

                seg = content[offset:offset+length]

            segments.append(seg)

            offset += length

        return segments
