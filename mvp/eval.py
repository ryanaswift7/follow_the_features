import gc
import json

import evaluate
import torch
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# Unified eval_and_save function for both baseline and trained model evals
def eval_and_save(
    model,
    tokenizer,
    model_type,
    model_id,
    config,
    out_dir,
    top_k=None,
    threshold=None,
    model_path=None,
    dataset_path=None,
    include_sft=False,
):
    import json

    from mvp.utils.experiment_helpers import config_hash, ensure_dir

    ensure_dir(out_dir)
    datasets_map = load_eval_datasets(seed=config.seed, limit=config.eval.num_samples)
    rouge = evaluate.load("rouge")
    batch_size = getattr(config.eval, "batch_size", 4)
    model_scores = {}
    for ds_name, data in datasets_map.items():
        preds, refs = generate_responses(
            model,
            tokenizer,
            data,
            batch_size=batch_size,
            max_new_tokens=config.eval.max_new_tokens,
        )
        scores = rouge.compute(predictions=preds, references=refs, use_aggregator=True)
        model_scores[ds_name] = round(scores["rougeL"] * 100, 2)
    eval_score = sum(model_scores.values()) / len(model_scores) if model_scores else 0
    eval_scores = {"average_eval_score": eval_score, **model_scores}
    # Additional metadata for aggregation/plotting
    import os

    # Determine dataset path and size for trained models
    # pruned_path = getattr(config, "pruned_path", None)
    # dataset_path = pruned_path if pruned_path else getattr(config, "dataset_path", None)
    # model_path = getattr(config, "model_path", None)
    dataset_size = None
    if dataset_path and os.path.exists(dataset_path):
        from mvp.utils.experiment_helpers import get_dataset_size

        dataset_size = get_dataset_size(dataset_path)
    result = {
        "model_type": model_type,
        "model_id": model_id,
        "top_k": top_k,
        "threshold": threshold,
        "dataset_size": dataset_size,
        "model_path": model_path,
        "dataset_path": dataset_path,
        "eval_config": config.eval.dict()
        if hasattr(config.eval, "dict")
        else dict(config.eval),
        "prune_config": config.prune.dict()
        if hasattr(config.prune, "dict")
        else dict(config.prune),
        "sft_config": config.sft.dict()
        if hasattr(config, "sft") and hasattr(config.sft, "dict")
        else None,
        "prune_id": config_hash(
            config.prune.dict() if hasattr(config.prune, "dict") else dict(config.prune)
        ),
        "sft_id": config_hash(config.sft.dict())
        if hasattr(config, "sft") and hasattr(config.sft, "dict")
        else None,
        "eval_id": config_hash(
            config.eval.dict() if hasattr(config.eval, "dict") else dict(config.eval)
        ),
        "eval_scores": eval_scores,
    }
    if include_sft and hasattr(config, "sft") and hasattr(config.sft, "dict"):
        result["sft_id"] = config_hash(config.sft.dict())
        result["sft_config"] = config.sft.dict()
    eval_config_path = os.path.join(out_dir, "eval_config.json")
    with open(eval_config_path, "w") as f:
        json.dump(config.eval.dict(), f, indent=4)
    results_path = os.path.join(out_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(result, f, indent=4)
    return result


## All configuration is now loaded from the YAML config file via the config object passed to eval_run


# ================= DATASET LOADERS =================
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
    # ds = ds.select(range(len(ds) - 1000, len(ds)))  # Take last 1000 as pseudo-test
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

    return eval_data


# ================= EVALUATION LOGIC =================


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


def eval_run(
    config,
    adapter_path,
    result_dir,
    dataset_path,
    model_path,
    top_k,
    threshold,
):
    import torch

    base_model_id = config.base_model_id
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_id)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, device_map="auto", torch_dtype=torch.bfloat16
    )
    print(f"Loading Adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    # Set model_id to the lowest directory name in adapter_path
    import os

    model_id = os.path.basename(adapter_path.rstrip("/"))
    result = eval_and_save(
        model,
        tokenizer,
        model_type="adapter_model",
        model_id=model_id,
        config=config,
        out_dir=result_dir,
        top_k=top_k,
        threshold=threshold,
        model_path=model_path,
        dataset_path=dataset_path,
        include_sft=True,
    )
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_baseline_evals(config, result_dir):
    """
    Run and save baseline (base and prune) model evaluations ONCE per experiment sweep.
    """
    import os

    import torch

    from mvp.utils.experiment_helpers import config_hash, ensure_dir

    base_model_id = config.base_model_id
    prune_model_id = getattr(config.prune, "model_id", None)
    results = {}

    eval_hash = config_hash(config.eval.dict())

    # Use get_exp_paths to get canonical model and eval dirs for baselines
    from mvp.utils.experiment_paths import get_exp_paths

    # Use top_k=0, threshold=0 as dummy values for baselines
    (
        _results_dir,
        _dataset_dir,
        models_dir,
        _pruned_path,
        _model_subdir,
        eval_subdir,
        _timestamp,
    ) = get_exp_paths(result_dir, 0, 0, config)
    # Base model dir: models/base_model/<base_model_id>/<eval_id>
    base_model_id = config.base_model_id
    eval_config_dict = (
        config.eval.dict() if hasattr(config.eval, "dict") else dict(config.eval)
    )
    from mvp.utils.experiment_helpers import config_hash

    eval_id = f"eval_{config_hash(eval_config_dict)}"
    base_model_dir = os.path.join(
        models_dir, "base_model", base_model_id.replace("/", "_"), eval_id
    )
    ensure_dir(base_model_dir)

    # Base model
    base_results_path = os.path.join(base_model_dir, "eval_results.json")
    base_valid = False
    base_row = None
    if os.path.exists(base_results_path):
        try:
            with open(base_results_path, "r") as f:
                loaded = json.load(f)
            # Unwrap if wrapped as {"eval_scores": {...}}
            if isinstance(loaded, dict) and list(loaded.keys()) == ["eval_scores"]:
                base_row = loaded["eval_scores"]
            else:
                base_row = loaded
            base_valid = base_row.get("eval_id") == eval_hash
        except json.JSONDecodeError:
            print(
                f"[BASELINE] Warning: Could not decode JSON in {base_results_path}. Will re-run baseline eval."
            )
            base_valid = False
    if not base_valid:
        print("Evaluating base model (no adapter)...")
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_id)
        tokenizer.padding_side = "left"
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id, device_map="auto", torch_dtype=torch.bfloat16
        )
        results["base_model"] = eval_and_save(
            base_model,
            tokenizer,
            model_type="base_model",
            model_id=base_model_id,
            config=config,
            out_dir=base_model_dir,
            include_sft=False,
        )
        del base_model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print(
            "[BASELINE] Base model eval already exists and matches eval_id, skipping."
        )
        results["base_model"] = {"eval_scores": base_row}

    # Prune model
    if prune_model_id:
        # Prune model dir: models/prune_model/<prune_model_id>/<eval_id>
        prune_model_dir = os.path.join(
            models_dir, "prune_model", prune_model_id.replace("/", "_"), eval_id
        )
        ensure_dir(prune_model_dir)
        prune_results_path = os.path.join(prune_model_dir, "eval_results.json")
        prune_valid = False
        prune_row = None
        if os.path.exists(prune_results_path):
            try:
                with open(prune_results_path, "r") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict) and list(loaded.keys()) == ["eval_scores"]:
                    prune_row = loaded["eval_scores"]
                else:
                    prune_row = loaded
                prune_valid = prune_row.get("eval_id") == eval_hash
            except json.JSONDecodeError:
                print(
                    f"[BASELINE] Warning: Could not decode JSON in {prune_results_path}. Will re-run prune baseline eval."
                )
                prune_valid = False
        if not prune_valid:
            print("Evaluating prune model (no adapter)...")
            tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_id)
            tokenizer.padding_side = "left"
            prune_model = AutoModelForCausalLM.from_pretrained(
                prune_model_id, device_map="auto", torch_dtype=torch.bfloat16
            )
            results["prune_model"] = eval_and_save(
                prune_model,
                tokenizer,
                model_type="prune_model",
                model_id=prune_model_id,
                config=config,
                out_dir=prune_model_dir,
                include_sft=False,
            )
            del prune_model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
        else:
            print(
                "[BASELINE] Prune model eval already exists and matches eval_id, skipping."
            )
            results["prune_model"] = {"eval_scores": prune_row}
    return results
