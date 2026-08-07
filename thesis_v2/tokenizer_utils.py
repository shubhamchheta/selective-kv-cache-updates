from transformers import AutoTokenizer
from dotenv import load_dotenv
import os

from config import MODEL_NAME

load_dotenv("../.env")


def load_tokenizer():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        token=os.getenv("HF_TOKEN")
    )

    return tokenizer
