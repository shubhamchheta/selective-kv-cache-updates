import torch

from selective_kvcache import generate
from cag import similarity


class Evaluator:

    def __init__(self, model, tokenizer):

        self.model = model
        self.tokenizer = tokenizer

    def ask(self, question, cache):

        prompt = f"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You must answer ONLY using the cached textbook.

If the answer is not explicitly present in the textbook,
reply exactly:

I don't know.

Do not use your own knowledge.
Do not guess.
Answer briefly.

<|eot_id|>

<|start_header_id|>user<|end_header_id|>

Question:
{question}

<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
"""

        input_ids = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).input_ids

        output = generate(
            self.model,
            input_ids,
            cache
        )

        answer = self.tokenizer.decode(
            output[0],
            skip_special_tokens=True
        )

        return answer.strip()

    def evaluate(
        self,
        questions,
        full_cache,
        selective_cache
    ):

        full_scores = []
        selective_scores = []

        for question, gt in questions:

            full = self.ask(
                question,
                full_cache
            )

            selective = self.ask(
                question,
                selective_cache
            )

            full_score = similarity.bert(
                full,
                gt
            )

            selective_score = similarity.bert(
                selective,
                gt
            )

            full_scores.append(full_score)
            selective_scores.append(selective_score)

        return (
            sum(full_scores) / len(full_scores),
            sum(selective_scores) / len(selective_scores)
        )
