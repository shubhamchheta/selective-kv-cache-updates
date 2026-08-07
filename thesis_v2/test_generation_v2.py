import copy
import sys

sys.path.append("..")

from tokenizer_utils import load_tokenizer
from cache_builder import CacheBuilder
from cache_mapper import CacheMapper
from cache_truncate import CacheTruncate
from cache_merge import CacheMerge
from tail_builder import TailBuilder

from dataset import knowledge, segments

from cache_adapter import CacheAdapter

import torch


def generate(
    model,
    input_ids,
    past_key_values,
    max_new_tokens=30,
):

    device = model.model.embed_tokens.weight.device

    input_ids = input_ids.to(device)

    origin = input_ids.clone()

    next_token = input_ids

    output = input_ids.clone()

    with torch.no_grad():

        for _ in range(max_new_tokens):

            outputs = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True
            )

            logits = outputs.logits[:, -1, :]

            next_token = logits.argmax(dim=-1).unsqueeze(-1)

            next_token = next_token.to(device)

            past_key_values = outputs.past_key_values

            output = torch.cat(
                (output, next_token),
                dim=1
            )

            if next_token.item() == model.config.eos_token_id:
                break

    return output[:, origin.shape[-1]:]

print("Loading tokenizer...")
tokenizer = load_tokenizer()

builder = CacheBuilder()
mapper = CacheMapper(tokenizer)

####################################################
# WHOLE CACHE
####################################################

print("\nBuilding Whole Cache")

whole_ids = tokenizer.encode(
    knowledge,
    return_tensors="pt"
)

whole_cache = builder.build(whole_ids)

####################################################
# UPDATE CACHE
####################################################

changed = 1

mapping = mapper.build_mapping()

keep = mapping[changed]["start"]

prefix_cache = copy.deepcopy(whole_cache)

prefix_cache = CacheTruncate.truncate(
    prefix_cache,
    keep
)

tail = TailBuilder.build(
    segments,
    changed
)

tail_ids = tokenizer.encode(
    tail,
    return_tensors="pt"
)

tail_cache = builder.build(tail_ids)

updated_cache = CacheMerge.merge(
    prefix_cache,
    tail_cache
)

####################################################
# QUESTION
####################################################

question = """
Answer ONLY from the document.

Question:
How many legs do dogs have?

Answer:
"""

question_ids = tokenizer.encode(
    question,
    return_tensors="pt"
)

####################################################
# WHOLE CACHE
####################################################

print("\n" + "="*60)
print("WHOLE CACHE")
print("="*60)

answer = generate(
    builder.model,
    question_ids,
    whole_cache,
    max_new_tokens=30
)

print(
    tokenizer.decode(
        answer[0],
        skip_special_tokens=True
    )
)

####################################################
# UPDATED CACHE
####################################################

print("\n" + "="*60)
print("UPDATED CACHE")
print("="*60)

answer = generate(
    builder.model,
    question_ids,
    updated_cache,
    max_new_tokens=30
)

print(
    tokenizer.decode(
        answer[0],
        skip_special_tokens=True
    )
)
