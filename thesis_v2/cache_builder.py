import os
import sys
import time
sys.path.append("..")

import torch

from dotenv import load_dotenv
from transformers import AutoModelForCausalLM

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

load_dotenv("../.env")


class CacheBuilder:

    def __init__(self):

        print("Loading model...")

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            token=os.getenv("HF_TOKEN")
        )

        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            token=os.getenv("HF_TOKEN")
        )

        print("Model Loaded.")



    def build(self, input_ids, past_key_values=None):

        device = self.model.model.embed_tokens.weight.device

        input_ids = input_ids.to(device)
        
        with torch.no_grad():
            t1 = time.time()
            outputs = self.model(
                input_ids=input_ids,
                use_cache=True,
        		past_key_values=past_key_values
                
            )
            print(type(past_key_values))
            if past_key_values is not None:
                    print(len(past_key_values))
            torch.cuda.synchronize()

            t2 = time.time()

        print("Model only :", t2 - t1)

        return outputs.past_key_values

    def build_text(self, text, past_key_values=None):
        

        t1 = time.time()
        
        input_ids = self.tokenizer(
        	text,
        	return_tensors="pt",
		add_special_tokens=False
        ).input_ids
        t2 = time.time()
        cache = self.build(
            input_ids,
            past_key_values=past_key_values
        )
        t3 = time.time()
        print("--------------------------------")
        print("Input Tokens :", input_ids.shape[1])
        print("Tokenization :", round(t2 - t1, 6))
        print("Forward Pass :", round(t3 - t2, 6))
        print("Total        :", round(t3 - t1, 6))

        return cache

        
       # return self.build(input_ids,past_key_values=past_key_values)
