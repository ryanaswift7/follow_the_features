import gc
import json
import os
from pathlib import Path
from typing import Optional, cast

import torch
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.suite_cfg_schema import SuiteConfig
from src.utils.constants import (
    EVAL_CONFIG_FILENAME,
    EVAL_ID_KEY,
    EVAL_RESULTS_FILENAME,
    EVAL_RESULTS_KEY,
)
from src.utils.enums import ModelType
from src.utils.evaluator import Evaluator
from src.utils.ollama_manager import OllamaManager


# Unified eval_and_save function for both baseline and trained model evals
def eval_and_save(
    model,
    tokenizer,
    suite: SuiteConfig,
    model_type: ModelType,
    dataset_size: Optional[int] = None,
    topk: Optional[int] = None,
):
    datasets_map = load_eval_datasets(seed=suite.seed, limit=suite.eval.cfg.num_samples)
    batch_size = suite.eval.params.batch_size
    model_scores = {}

    with OllamaManager(ollama_judge=suite.eval.cfg.ollama_judge) as ol_man:
        evaluator = Evaluator(
            model=model,
            tokenizer=tokenizer,
            ollama_judge=suite.eval.cfg.ollama_judge,
        )
        for ds_name, data in datasets_map.items():
            preds, refs = generate_responses(
                model,
                tokenizer,
                data,
                batch_size=batch_size,
                max_new_tokens=suite.eval.cfg.max_new_tokens,
            )
            eval_payload = []
            for i in range(len(preds)):
                eval_payload.append(
                    {
                        "prompt": data[i]["prompt"],
                        "reference": refs[i],
                        "generation": preds[i],
                    }
                )

            dataset_means = evaluator.run_full_eval(
                test_data=eval_payload, port=ol_man.port
            )

            model_scores[ds_name] = {
                "rougeL": round(dataset_means["rouge_l"] * 100, 2),
                "perplexity": round(dataset_means["perplexity"], 2),
                "judge_score": round(dataset_means["judge_score"], 3),
            }

    # Calculate global average (using judge_score as the primary 'eval_score')
    if model_scores:
        avg_judge = sum(m["judge_score"] for m in model_scores.values()) / len(
            model_scores
        )
        eval_scores = {"average_eval_score": round(avg_judge, 3), **model_scores}
    else:
        eval_scores = {"average_eval_score": 0}

    cfg = {
        "model_type": model_type.value,
        "model_id": suite.sft.cfg.base_model_id
        if model_type != ModelType.SUPERVISOR
        else suite.prune.cfg.model_id,
        "topk": topk,
        "dataset_size": dataset_size,
        "model_path": suite.dirs.get_trained_model_root(
            dataset_size=cast(int, dataset_size), topk=cast(int, topk)
        ).as_posix()
        if model_type == ModelType.ADAPTER
        else None,
        "dataset_path": suite.dirs.get_dataset_path(
            dataset_size=cast(int, dataset_size), topk=cast(int, topk)
        ).as_posix()
        if model_type == ModelType.ADAPTER
        else None,
        "prune_id": suite.ids.prune,
        "sft_id": suite.ids.sft,
        "eval_id": suite.ids.eval,
        "prune_config": suite.prune.cfg.model_dump(),
        "sft_config": suite.sft.cfg.model_dump(),
        "eval_config": suite.eval.cfg.model_dump(),
    }
    results = {
        "eval_scores": eval_scores,
    }

    if model_type == ModelType.ADAPTER:
        eval_dir = suite.dirs.get_trained_model_eval(
            dataset_size=cast(int, dataset_size), topk=cast(int, topk)
        )
    elif model_type == ModelType.BASE:
        eval_dir = suite.dirs.base_model_eval
    elif model_type == ModelType.SUPERVISOR:
        eval_dir = suite.dirs.supervisor_model_eval

    eval_config_path = eval_dir / EVAL_CONFIG_FILENAME
    results_path = eval_dir / EVAL_RESULTS_FILENAME
    with open(eval_config_path, "w") as f:
        json.dump(cfg, f, indent=4)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    return


def load_eval_datasets(seed: int, limit=None):
    """
    Loads and normalizes 5 instruction datasets into a standard format:
    [{'prompt': str, 'reference': str, 'source': str}, ...]
    """
    eval_data = {}
    print("Loading Evaluation Datasets...")

    # 1. Dolly Eval (databricks-dolly-15k)
    # No official test split, so we take a random slice from the end
    print(" - Loading Dolly-15k...")
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    ds = ds.shuffle(seed=seed)
    if limit and limit < len(ds):
        ds = ds.select(range(limit))

    eval_data["DollyEval"] = [
        {
            "prompt": f"{row['instruction']}\n{row['context']}".strip(),
            "reference": row["response"],
        }
        for row in ds
    ]

    # 2. Self-Instruct (human_eval subset
    print(" - Loading Self-Instruct...")
    ds = load_dataset(
        "HuggingFaceH4/self-instruct-eval",
        split="test",
    )
    ds = ds.shuffle(seed=seed)
    if limit and limit < len(ds):
        ds = ds.select(range(limit))

    eval_data["SelfInst"] = [
        {
            "prompt": "Instruction: "
            + row["instruction"]
            + "\nInput: "
            + row["instances"][0]["input"],
            "reference": "Output: " + row["instances"][0]["output"],
        }
        for row in ds
    ]

    # 3. Vicuna Eval (Using GPT-4 answers as reference for Rouge)
    print(" - Loading Vicuna Eval...")
    ds = load_dataset("zhengxuanzenwu/vicuna-eval-with-gpt4", split="test")
    ds = ds.shuffle(seed=seed)
    if limit and limit < len(ds):
        ds = ds.select(range(limit))

    eval_data["VicunaEval"] = [
        {
            "prompt": row["instruction"],
            "reference": row["response"],  # GPT-4 response as gold standard
        }
        for row in ds
    ]

    # 4. Super-NaturalInstructions (S-NI)
    print(" - Loading Super-NaturalInstructions (S-NI)...")
    ds = load_dataset(
        "Muennighoff/natural-instructions",
        revision="refs/convert/parquet",
        split="default",
        data_files={"default": "default/validation/*.parquet"},
    )
    ds = ds.shuffle(seed=seed)
    if limit and limit < len(ds):
        ds = ds.select(range(limit))

    eval_data["S-NI"] = [
        {
            "prompt": "Instruction: "
            + row["definition"]
            + "\nPassage: "
            + row["inputs"],
            "reference": row["targets"],
        }
        for row in ds
    ]

    # 5. WizardLM (Evol-Instruct)
    print(" - Loading WizardLM (Evol-Instruct)...")
    ds = load_dataset(
        "WizardLM/WizardLM_evol_instruct_70k",
        split="train",
    )
    ds = ds.shuffle(seed=seed)
    if limit and limit < len(ds):
        ds = ds.select(range(limit))

    eval_data["WizardLM"] = [
        {
            "prompt": row["instruction"],
            "reference": row["output"],
        }
        for row in ds
    ]

    return eval_data


def generate_responses(model, tokenizer, dataset, batch_size, max_new_tokens):
    """Generates responses for a dataset."""
    model.eval()
    predictions = []

    # Simple batching
    prompts = [x["prompt"] for x in dataset]
    refs = [x["reference"] for x in dataset]

    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch_prompts = prompts[i : i + batch_size]

        # Apply Chat Template (Simple user/assistant format)
        formatted_prompts = []
        for p in batch_prompts:
            msgs = [{"role": "user", "content": p}]
            formatted_prompts.append(
                tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
            )

        inputs = tokenizer(
            formatted_prompts, return_tensors="pt", padding=True, truncation=True
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy decoding for deterministic eval
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode and strip the prompt
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for j, text in enumerate(decoded):
            # Heuristic to strip input prompt if it leaks into output
            # (Gemma chat template usually handles this, but we ensure cleanliness)
            response = text.split("model")[-1] if "model" in text else text
            predictions.append(response.strip())

    return predictions, refs


def run_eval(
    suite: SuiteConfig,
    model_type: ModelType,
    dataset_size: Optional[int] = None,
    topk: Optional[int] = None,
):
    if model_type == ModelType.SUPERVISOR:
        model_id = suite.prune.cfg.model_id
    else:
        model_id = suite.sft.cfg.base_model_id

    tokenizer = AutoTokenizer.from_pretrained(suite.sft.cfg.tokenizer_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.bfloat16
    )

    if model_type == ModelType.ADAPTER:
        assert dataset_size is not None and topk is not None, (
            "dataset_size and topk must be provided for adapter evals"
        )
        adapter_path = suite.dirs.get_trained_model_root(
            dataset_size=cast(int, dataset_size), topk=cast(int, topk)
        ).as_posix()
        print(f"Loading Adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()

    eval_and_save(
        model=model,
        tokenizer=tokenizer,
        suite=suite,
        model_type=model_type,
        dataset_size=dataset_size,
        topk=topk,
    )
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return


def verify_or_run_eval(
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
    tokenizer = AutoTokenizer.from_pretrained(suite.sft.cfg.tokenizer_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.bfloat16
    )

    eval_and_save(
        model=model,
        tokenizer=tokenizer,
        model_type=model_type,
        suite=suite,
        dataset_size=None,
        topk=None,
    )

    # 4. Cleanup
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return


def run_baseline_evals(suite: SuiteConfig):
    """
    Run and save baseline (base and supervisor) model evaluations.
    """
    # Base Model
    verify_or_run_eval(
        model_id=suite.sft.cfg.base_model_id,
        model_type=ModelType.BASE,
        eval_dir=suite.dirs.base_model_eval,
        suite=suite,
    )

    # Supervisor Model
    verify_or_run_eval(
        model_id=suite.prune.cfg.model_id,
        model_type=ModelType.SUPERVISOR,
        eval_dir=suite.dirs.supervisor_model_eval,
        suite=suite,
    )

    return
