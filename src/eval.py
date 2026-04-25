import gc
import json
import os
from pathlib import Path
from typing import Optional, cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.reference_models import generate_reference_models
from src.schemas.suite import SuiteConfig
from src.utils.constants import (
    EVAL_CONFIG_FILENAME,
    EVAL_ID_KEY,
    EVAL_RESULTS_FILENAME,
    EVAL_RESULTS_KEY,
)
from src.utils.enums import ModelType
from src.utils.evals import eval_and_save
from src.utils.ensure_pad_token import ensure_pad_token


# def run_eval(
#     suite: SuiteConfig,
#     model_type: ModelType,
#     dataset_size: Optional[int] = None,
#     topk: Optional[int] = None,
# ):
#     model_id = suite.sft.cfg.base_model_id


#     if model_type == ModelType.ADAPTER:
#         assert dataset_size is not None and topk is not None, (
#             "dataset_size and topk must be provided for adapter evals"
#         )
#         adapter_path = suite.dirs.get_trained_model_root(
#             dataset_size=cast(int, dataset_size), topk=cast(int, topk)
#         ).as_posix()
#         print(f"Loading Adapter: {adapter_path}")
#         model = AutoModelForCausalLM.from_pretrained(adapter_path, device_map="auto",  dtype=torch.bfloat16)
#         # model = model.merge_and_unload()
#         tokenizer = AutoTokenizer.from_pretrained(suite.sft.cfg.tokenizer_id, fix_mistral_regex=True)

#     else:
#         # need to enable mistral tokenizer fix to use tokenizer in vllm
#         tokenizer = AutoTokenizer.from_pretrained(
#             suite.sft.cfg.tokenizer_id, fix_mistral_regex=True
#         )
#         model = AutoModelForCausalLM.from_pretrained(
#             model_id, device_map="auto", dtype=torch.bfloat16
#         )

def run_eval(
    suite: SuiteConfig,
    model_type: ModelType,
    dataset_size: int,
    topk: int
):
    assert dataset_size is not None and topk is not None, (
        "dataset_size and topk must be provided for adapter evals"
    )
    adapter_path = suite.dirs.get_trained_model_root(
        dataset_size=cast(int, dataset_size),
        topk=cast(int, topk),
    ).as_posix()
    gen_cache_dir = suite.dirs.get_trained_model_gen_cache(
        dataset_size=dataset_size,
        topk=topk,
        max_new_tokens=suite.eval.cfg.max_new_tokens,
    )
    print(f"Loading Adapter: {adapter_path}")
    model = AutoModelForCausalLM.from_pretrained(adapter_path, device_map="auto",  dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(suite.sft.cfg.tokenizer_id, fix_mistral_regex=True)
    model, tokenizer = ensure_pad_token(model=model, tokenizer=tokenizer)

    eval_and_save(
        model=model,
        tokenizer=tokenizer,
        suite=suite,
        model_type=model_type,
        dataset_size=dataset_size,
        topk=topk,
        gen_cache_dir=gen_cache_dir,
    )
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return


def verify_or_run_baseline(
    model_id: str, model_type: ModelType, eval_dir: Path, suite: SuiteConfig
) -> None:
    """
    Checks for existing valid evaluations or runs a new one.
    """
    cfg_path = eval_dir / EVAL_CONFIG_FILENAME
    results_path = eval_dir / EVAL_RESULTS_FILENAME
    valid = False
    row = None

    # 1. Validate existing config/results
    if os.path.exists(cfg_path) and os.path.exists(results_path):
        try:
            with open(cfg_path, "r") as f:
                cfg_data = json.load(f)
            with open(results_path, "r") as f:
                res_data = json.load(f)

            # Check if IDs match
            if cfg_data.get(EVAL_ID_KEY) == suite.ids.eval:
                valid = True
                # Unwrap results if needed
                row = (
                    res_data.get(EVAL_RESULTS_KEY, res_data)
                    if isinstance(res_data, dict)
                    else res_data
                )
        except (json.JSONDecodeError, IOError) as e:
            print(
                f"[BASELINE] Warning: Error reading {model_type.name} cache in {eval_dir}: {e}"
            )

    # 2. Return cached if valid
    if valid and row is not None:
        print(
            f"[BASELINE] {model_type.name} eval already exists and matches eval_id, skipping."
        )
        return

    # 3. Otherwise, Run Eval
    print(f"Evaluating {model_type.name} model ({model_id})...")

    if model_type == ModelType.BASE:
        tokenizer = AutoTokenizer.from_pretrained(suite.sft.cfg.tokenizer_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.bfloat16
        )
        model, tokenizer = ensure_pad_token(model=model, tokenizer=tokenizer)

        eval_and_save(
            model=model,
            tokenizer=tokenizer,
            model_type=model_type,
            suite=suite,
            dataset_size=None,
            topk=None,
            gen_cache_dir=suite.dirs.get_base_model_gen_cache(
                max_new_tokens=suite.eval.cfg.max_new_tokens),
        )

        # 4. Cleanup
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    else:
        generate_reference_models(suite=suite)

    return


def run_baseline_evals(suite: SuiteConfig):
    """
    Run and save baseline (base and reference) model evaluations.
    """
    # must run ref models first so that they are initialized for base model judge scores
    verify_or_run_baseline(
        model_id=suite.sft.cfg.base_model_id,
        model_type=ModelType.REFERENCE,
        eval_dir=suite.dirs.reference_model_eval,
        suite=suite,
    )
    verify_or_run_baseline(
        model_id=suite.sft.cfg.base_model_id,
        model_type=ModelType.BASE,
        eval_dir=suite.dirs.base_model_eval,
        suite=suite,
    )
    return
