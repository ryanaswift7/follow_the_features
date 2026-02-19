import gc
import json
from typing import Dict, Optional, Tuple, cast

import lm_eval
import torch
from datasets import load_dataset
from lm_eval.models.huggingface import HFLM
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM

from src.schemas.suite import SuiteConfig
from src.utils.constants import (
    EVAL_CONFIG_FILENAME,
    EVAL_RESULTS_FILENAME,
    INVALID_JUDGE_LOG_FILENAME,
)
from src.utils.enums import ModelType
from src.utils.evaluator import Evaluator
from src.utils.ollama_manager import OllamaManager


def write_eval_results_and_config(
    eval_scores: Dict,
    suite: SuiteConfig,
    model_type: ModelType,
    dataset_size: Optional[int] = None,
    topk: Optional[int] = None,
    ref_model_seed: Optional[int] = None,
):
    cfg = {
        "model_type": model_type.value,
        "model_id": suite.sft.cfg.base_model_id,
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
    # results = {
    #     "eval_scores": eval_scores,
    # }

    if model_type == ModelType.ADAPTER:
        eval_dir = suite.dirs.get_trained_model_eval(
            dataset_size=cast(int, dataset_size), topk=cast(int, topk)
        )
    elif model_type == ModelType.BASE:
        eval_dir = suite.dirs.base_model_eval
    elif model_type == ModelType.REFERENCE:
        assert ref_model_seed, (
            "Seed must be provided when running evals on reference models."
        )
        eval_dir = suite.dirs.get_ref_model_by_seed(ref_model_seed)

    eval_config_path = eval_dir / EVAL_CONFIG_FILENAME
    results_path = eval_dir / EVAL_RESULTS_FILENAME
    with open(eval_config_path, "w") as f:
        json.dump(cfg, f, indent=4)
    with open(results_path, "w") as f:
        json.dump(eval_scores, f, indent=4)
    return cfg


def eval_and_save(
    model,
    tokenizer,
    suite: SuiteConfig,
    model_type: ModelType,
    dataset_size: Optional[int] = None,
    topk: Optional[int] = None,
    save_results_and_config: bool = True,
    ref_model_seed: Optional[int] = None,
) -> Tuple[dict, Optional[dict]]:

    batch_size = suite.eval.params.batch_size
    model_scores = {}

    model.eval()
    torch.cuda.empty_cache()
    gc.collect()

    eval_scores = {}

    # doesnt make sense to run ref vs ref evals
    if model_type != ModelType.REFERENCE:
        datasets_map = load_eval_datasets(
            seed=suite.seed, limit=suite.eval.cfg.num_samples
        )
        with OllamaManager(ollama_judge=suite.eval.cfg.ollama_judge) as ol_man:
            evaluator = Evaluator(
                model=model,
                tokenizer=tokenizer,
                ollama_judge=suite.eval.cfg.ollama_judge,
            )

            for ds_name, data in datasets_map.items():
                # 1. Generate responses from the Candidate Model (the one being evaluated)
                candidate_preds, metric_refs = generate_responses(
                    model,
                    tokenizer,
                    data,
                    batch_size=batch_size,
                    max_new_tokens=suite.eval.cfg.max_new_tokens,
                )

                # 2. Generate responses from all Reference Models (Baselines)
                # We store these as a list of lists.
                # ref_model_outputs[0] = list of responses from ref_model_0
                ref_model_outputs = []

                for i in range(suite.eval.cfg.num_ref_models):
                    # Load the reference model dynamically
                    ref_path = suite.dirs.get_ref_model_by_seed(suite.seed + i)
                    print(
                        f"Generating baselines for {ds_name} using ref model: {ref_path}"
                    )

                    ref_model = AutoModelForCausalLM.from_pretrained(
                        ref_path, device_map="auto", torch_dtype=torch.bfloat16
                    )

                    # Generate responses using the ref model
                    # Note: We assume the tokenizer is compatible. If not, load ref_tokenizer here.
                    ref_preds, _ = generate_responses(
                        ref_model,
                        tokenizer,
                        data,
                        batch_size=batch_size,
                        max_new_tokens=suite.eval.cfg.max_new_tokens,
                    )

                    ref_model_outputs.append(ref_preds)

                    # Cleanup to free VRAM for the next ref model or evaluation
                    del ref_model
                    torch.cuda.empty_cache()

                # 3. Construct the Payload
                eval_payload = []
                for i in range(len(candidate_preds)):
                    # Aggregate the i-th response from every reference model into a single list
                    # This list corresponds to 'reference_judges' in the Evaluator
                    refs_for_this_sample = [run[i] for run in ref_model_outputs]

                    eval_payload.append(
                        {
                            "prompt": data[i]["prompt"],
                            "generation": candidate_preds[i],  # Candidate text
                            "reference_metrics": metric_refs[
                                i
                            ],  # Ground truth (for ROUGE)
                            "reference_judges": refs_for_this_sample,  # List of Ref Model texts (for Judge)
                        }
                    )

                # 4. Run Evaluation
                if model_type == ModelType.BASE:
                    log_file = suite.dirs.base_model_eval / INVALID_JUDGE_LOG_FILENAME
                elif model_type == ModelType.ADAPTER:
                    log_file = (
                        suite.dirs.get_trained_model_eval(
                            dataset_size=cast(int, dataset_size), topk=cast(int, topk)
                        )
                        / INVALID_JUDGE_LOG_FILENAME
                    )
                else:
                    raise ValueError(
                        f"Invalid model type {model_type.value.upper()} for judge evaluation."
                    )
                dataset_means = evaluator.run_full_eval(
                    test_data=eval_payload, port=ol_man.port, log_file=log_file
                )

                model_scores[ds_name] = {
                    "rougeL": round(dataset_means["rouge_l"] * 100, 2),
                    "perplexity": round(dataset_means["perplexity"], 2),
                    "judge_score": round(dataset_means["judge_score"], 3),
                    "invalid_judge_responses": dataset_means["invalid_judge_responses"],
                }

        # Calculate global average (using judge_score as the primary 'eval_score')
        if model_scores:
            avg_judge = sum(m["judge_score"] for m in model_scores.values()) / len(
                model_scores
            )
            ref_scores = {"average_judge_score": round(avg_judge, 3), **model_scores}
        else:
            ref_scores = {"average_judge_score": 0}

        eval_scores["reference_results"] = ref_scores

    # run lm evals
    lm_eval_results = {}
    hlfm_args = {
        "pretrained": model,
        "tokenizer": tokenizer,
        # "max_gen_tokens": suite.eval.cfg.max_new_tokens,
    }
    hflm_model = HFLM(**hlfm_args)

    raw_results = lm_eval.simple_evaluate(
        model=hflm_model,
        tasks=suite.eval.cfg.tasks,
        limit=suite.eval.cfg.task_limit,
        batch_size=suite.eval.params.batch_size,
        gen_kwargs={
            "max_gen_toks": suite.eval.cfg.max_new_tokens,
            "do_sample": False,
        },
    )

    # print(json.dumps(raw_results["results"], indent=4))

    metric_keys = [
        "acc_norm,none",
        "acc,none",
        "inst_level_strict_acc,none",
        "exact_match,none",
    ]

    accuracies = []
    for task_name, metrics in raw_results["results"].items():
        acc = None
        for key in metric_keys:
            val = metrics.get(key)
            if val is not None:
                acc = val
                break

        if acc is not None:
            lm_eval_results[task_name] = acc
            accuracies.append(acc)
        else:
            raise ValueError("Task result not extracted correctly")

    # lm_eval_results["average_lm_eval_score"] = sum(accuracies) / len(accuracies)
    selected_keys = [
        "leaderboard",
        "leaderboard_bbh",
        "leaderboard_mmlu_pro",
        "leaderboard_musr",
        "leaderboard_math_hard",
        "leaderboard_ifeval",
    ]
    filtered_lm_eval_results = {
        k: v for k, v in lm_eval_results.items() if k in selected_keys
    }
    # filtered_lm_eval_results["average_lm_eval_score"] = sum(
    #     list(filtered_lm_eval_results.values())
    # ) / len(filtered_lm_eval_results)
    eval_scores["lm_eval_results"] = filtered_lm_eval_results

    cfg = None
    if save_results_and_config:
        cfg = write_eval_results_and_config(
            eval_scores=eval_scores,
            suite=suite,
            model_type=model_type,
            dataset_size=dataset_size,
            topk=topk,
            ref_model_seed=ref_model_seed,
        )

    return eval_scores, cfg


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
