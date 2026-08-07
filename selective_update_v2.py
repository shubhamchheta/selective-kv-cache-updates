import torch
import os
import logging
import warnings
from thesis_v2.cache_builder import CacheBuilder
from thesis_v2.selective_updater import SelectiveUpdater

import textbook_dataset

from thesis_v2.evaluator import Evaluator
from change_detector import ChangeDetector

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

warnings.filterwarnings("ignore")
from transformers import logging as hf_logging

from thesis_v2.reporter import Reporter
from thesis_v2.cache_adapter import CacheAdapter

hf_logging.set_verbosity_error()
# ============================================================
# Load textbook datasets
# ============================================================

v1, questions = textbook_dataset.load("v1")
v2, _ = textbook_dataset.load("v2")


# ============================================================
# Detect first changed chapter
# ============================================================

changed = ChangeDetector.first_changed(v1, v2)

Reporter.header(changed)

builder = CacheBuilder()

Reporter.warmup()

builder.build_text("Warmup")

torch.cuda.synchronize()



updater = SelectiveUpdater(builder)

cache_v1, offline_time = updater.build_initial_cache(v1)


cache_v2, full_time = updater.build_full_cache(v2)


prefix_cache, prefix_tokens = updater.build_prefix_cache(changed)
suffix = "\n\n".join(v2[changed:])

updated_cache, selective_time = updater.build_selective_cache(
    suffix,
    prefix_cache
)

evaluator = Evaluator(
    builder.model,
    builder.tokenizer
)

full_score, selective_score = evaluator.evaluate(
    questions,
    cache_v2,
    updated_cache
)

print(CacheAdapter.get_keys(cache_v1,0).shape[2])
print(CacheAdapter.get_keys(cache_v2,0).shape[2])
print(CacheAdapter.get_keys(updated_cache,0).shape[2])
print(prefix_tokens)

Reporter.summary(
    full_score,
    selective_score,
    offline_time,
    full_time,
    selective_time,
    CacheAdapter.get_keys(cache_v1, 0).shape[2],
    CacheAdapter.get_keys(cache_v2, 0).shape[2],
    prefix_tokens,
    CacheAdapter.get_keys(updated_cache, 0).shape[2]
)