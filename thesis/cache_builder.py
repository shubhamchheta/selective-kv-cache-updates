import os
import torch

from dotenv import load_dotenv

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DynamicCache,
)

load_dotenv("../.env")

MODEL = "meta-llama/Llama-3.1-8B-Instruct"


class CacheBuilder:

    def __init__(self):

        print("Loading tokenizer...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL,
            token=os.getenv("HF_TOKEN")
        )

        print("Loading model...")

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            torch_dtype=torch.float16,
            device_map="auto",
            token=os.getenv("HF_TOKEN")
        )

        self.device = self.model.model.embed_tokens.weight.device

        print("Model Loaded.")

    def build_cache(self, token_ids):

        ids = torch.tensor([token_ids]).to(self.device)

        cache = DynamicCache()

        with torch.no_grad():

            outputs = self.model(
                input_ids=ids,
                past_key_values=cache,
                use_cache=True
            )

        return outputs.past_key_values
