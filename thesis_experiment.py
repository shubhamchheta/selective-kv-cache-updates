import json
import time

import textbook_dataset

from selective_kvcache import (
    prepare_kvcache,
    generate,
    MODEL_NAME
)

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from dotenv import load_dotenv

load_dotenv(".env")

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    token=os.getenv("HF_TOKEN")
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    token=os.getenv("HF_TOKEN")
)

print("Model loaded.")

def evaluate(cache, questions):

    results = []

    total_generate = 0

    for i, (question, answer) in enumerate(questions):

        t1 = time.time()

        response = generate(
            model,
            tokenizer,
            question,
            cache
        )

        t2 = time.time()

        total_generate += t2 - t1

        results.append({
            "id": i,
            "question": question,
            "expected": answer,
            "generated": response
        })

        print("=" * 60)
        print("Question", i + 1)
        print("=" * 60)
        print("Q :", question)
        print("GT:", answer)
        print("A :", response)
        print()

    return results, total_generate / len(questions)


def main():

    print("=" * 60)
    print("BASELINE")
    print("=" * 60)

    text_list, dataset = textbook_dataset.load("v1")

    questions = list(dataset)

    knowledge = "\n\n\n\n\n".join(text_list)

    t1 = time.time()

    cache, build_time = prepare_kvcache(
        knowledge,
        filepath="./data_cache/textbook_v1.pt"
    )

    t2 = time.time()

    results, avg_generate = evaluate(
        cache,
        questions
    )

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("Build Time :", build_time)
    print("Average Generation :", avg_generate)

    with open(
        "./results/textbook_baseline.json",
        "w"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )


if __name__ == "__main__":
    main()
