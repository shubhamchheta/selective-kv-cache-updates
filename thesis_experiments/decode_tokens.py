from transformers import AutoTokenizer
from dotenv import load_dotenv
import os

load_dotenv("../.env")

tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    token=os.getenv("HF_TOKEN")
)

for token in [382, 627]:
    print("Token ID:", token)
    print("Decoded :", repr(tokenizer.decode([token])))
    print("-" * 40)
