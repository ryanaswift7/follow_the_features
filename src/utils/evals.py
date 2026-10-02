import gc
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, cast

import lm_eval
from lm_eval.models.huggingface import HFLM
import torch
from accelerate import dispatch_model
from accelerate.hooks import remove_hook_from_module

from src.schemas.suite import SuiteConfig
from src.utils.alpaca_eval_runner import run_alpaca_eval_from_model
from src.utils.constants import (
    EVAL_CONFIG_FILENAME,
    EVAL_RESULTS_FILENAME,
)
from src.utils.enums import ModelType


def write_eval_results_and_config(
    raw_lm_eval_scores: Dict,
    eval_scores: Dict,
    suite: SuiteConfig,
    model_type: ModelType,
    dataset_size: Optional[int] = None,
    topk: Optional[int] = None,
    ref_model_seed: Optional[int] = None,
):
    if model_type == ModelType.ADAPTER:
        model_path = suite.dirs.get_trained_model_root(
            dataset_size=cast(int, dataset_size), topk=cast(int, topk)
        ).as_posix()
        dataset_path = suite.dirs.get_dataset_path(
            dataset_size=cast(int, dataset_size), topk=cast(int, topk)
        ).as_posix()
    elif model_type == ModelType.REFERENCE:
        assert ref_model_seed is not None, (
            "Seed must be provided when running evals on reference models."
        )
        model_path = suite.dirs.get_ref_model_root_by_seed(ref_model_seed).as_posix()
        dataset_path = None
    elif model_type == ModelType.BASE:
        model_path = suite.dirs.base_model_root.as_posix()
        dataset_path = None
    else:
        model_path = None
        dataset_path = None

    cfg = {
        "model_type": model_type.value,
        "model_id": suite.sft.cfg.base_model_id,
        "topk": topk,
        "dataset_size": dataset_size,
        "model_path": model_path,
        "dataset_path": dataset_path,
        "prune_id": suite.ids.prune,
        "sft_id": suite.ids.sft,
        "eval_id": suite.ids.eval,
        "prune_config": suite.prune.cfg.model_dump(),
        "sft_config": suite.sft.cfg.model_dump(),
        "eval_config": suite.eval.cfg.model_dump(),
        "seed": ref_model_seed if ref_model_seed is not None else suite.active_seed,
        "seeds": suite.seeds,
    }
    # results = {
    #     "eval_scores": eval_scores,
    # }

    if model_type == ModelType.ADAPTER:
        eval_dir = suite.dirs.get_trained_model_eval(
            dataset_size=cast(int, dataset_size), topk=cast(int, topk)
        )
    elif model_type == ModelType.BASE:
        eval_dir = suite.dirs.get_base_model_eval_by_seed(suite.seed)
    elif model_type == ModelType.REFERENCE:
        assert ref_model_seed is not None, (
            "Seed must be provided when running evals on reference models."
        )
        eval_dir = suite.dirs.get_ref_model_eval_by_seed(ref_model_seed)

    eval_config_path = eval_dir / EVAL_CONFIG_FILENAME
    results_path = eval_dir / EVAL_RESULTS_FILENAME
    raw_results_path = eval_dir / f"raw_{EVAL_RESULTS_FILENAME}"
    with open(eval_config_path, "w") as f:
        json.dump(cfg, f, indent=4)
    with open(results_path, "w") as f:
        json.dump(eval_scores, f, indent=4)
    with open(raw_results_path, "w") as f:
        json.dump(raw_lm_eval_scores, f, indent=4)
    return cfg


def run_phase1_lm_eval(model, tokenizer, suite, gen_cache_dir: Path) -> Tuple[Dict, Dict]:
    """Phase 1: LM Eval evaluation with full caching"""
    print("\n" + "="*60)
    print("PHASE 1: Starting LM Eval evaluation...")
    print("="*60)

    if not suite.eval.cfg.lm_eval.run:
        print("LM eval disabled in suite config. Skipping Phase 1.")
        return {}, {}
    
    if suite.eval.cfg.lm_eval.tasks is None:
        print("No LM Eval tasks specified in suite. Skipping Phase 1.")
        return {}, {}

    # Ensure the cache directory exists and define the cache path
    gen_cache_dir = Path(gen_cache_dir)
    gen_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = gen_cache_dir / "phase1_lm_eval.json"

    # --- CACHE CHECK ---
    selected_keys = {
        "leaderboard_ifeval": "ifeval",
        "leaderboard_bbh": "bbh",
        "leaderboard_math_hard": "math_hard",
        "leaderboard_gpqa": "gpqa",
        "leaderboard_musr": "musr",
        "leaderboard_mmlu_pro": "mmlu_pro",
    }
    expected_filtered_keys = {"leaderboard", *selected_keys.values()}

    if cache_path.exists():
        print(f"Found cached Phase 1 results at {cache_path}. Skipping evaluation.")
        with open(cache_path, "r") as f:
            cache_data = json.load(f)
        if cache_data.get("filter_version") == 2 and set(cache_data.get("filtered", {})) == expected_filtered_keys:
            return cache_data["filtered"], cache_data["raw"]
        print("Cached Phase 1 metric format is stale. Recomputing evaluation.")

    # --- EVALUATION ---
    lm_eval_results = {}
    raw_lm_eval_scores = {}

    need_restore_map = False
    if getattr(model, "hf_device_map", None) is not None:
        print("Preparing model for LM Eval: Saving device map and moving to cuda:0...")
        original_map = model.hf_device_map
        remove_hook_from_module(model, recurse=True)
        model.to("cuda:0")
        need_restore_map = True
    torch.cuda.empty_cache()
    gc.collect()

    print("Wrapping existing model in HFLM for lm_eval (reusing loaded VRAM)...")
    hflm_model = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=suite.eval.params.batch_size,
    )

    print(f"Running simple_evaluate on tasks: {suite.eval.cfg.lm_eval.tasks}...")


    # from transformers import GenerationConfig

    # inspect what the model is actually carrying
    hflm_model.model.generation_config.max_new_tokens = suite.eval.cfg.lm_eval.max_new_tokens
    # print(hflm_model.model.generation_config)

    # force a clean generation config
    # hflm_model.model.generation_config = GenerationConfig(max_new_tokens=256)

    raw_results = lm_eval.simple_evaluate(
        model=hflm_model, 
        tasks=suite.eval.cfg.lm_eval.tasks,
        limit=suite.eval.cfg.lm_eval.task_limit,
        batch_size=suite.eval.params.batch_size,
        gen_kwargs={"max_gen_toks": suite.eval.cfg.lm_eval.max_new_tokens},
        # if "olmo" not in suite.sft.cfg.base_model_id.lower() else 
        #     {"max_new_tokens": suite.eval.cfg.lm_eval.max_new_tokens,},
        apply_chat_template=True,
    )

    if need_restore_map:
        print("LM Eval complete. Restoring original multi-GPU device map...")
        dispatch_model(model, device_map=original_map)
    torch.cuda.empty_cache()
    gc.collect()

    print("Extracting and filtering evaluation metrics...")
    metric_lookups = {
        "acc_norm": ["acc_norm,none", "acc_norm"],
        "acc": ["acc,none", "acc"],
        "exact_match": ["exact_match,get-answer", "exact_match,remove_whitespace", "exact_match,none", "exact_match"],
        "f1": ["f1,none", "f1"],
        "inst_level_strict_acc": ["inst_level_strict_acc,none", "inst_level_strict_acc"]
    }

    print_labels = {
        "acc_norm": "Norm",
        "acc": "Acc",
        "exact_match": "EM",
        "f1": "F1",
        "inst_level_strict_acc": "Inst"
    }

    print("\n--- Phase 1 Task Results ---")
    for task_name, metrics in raw_results["results"].items():
        found_val = None
        found_name = None
        
        for base_metric, keys_to_try in metric_lookups.items():
            for key in keys_to_try:
                val = metrics.get(key)
                if val is not None:
                    found_val = val
                    found_name = base_metric
                    break 
            if found_val is not None:
                break 

        if found_val is None:
            raise ValueError(f"Task result not extracted correctly for: {task_name}")

        lm_eval_results[task_name] = found_val
        label = print_labels.get(found_name, found_name)
        print(f"  > {task_name:<20} | {label}: {found_val:.4f}")

    filtered_lm_eval_results = {
        output_key: lm_eval_results[input_key]
        for input_key, output_key in selected_keys.items()
        if input_key in lm_eval_results
    }

    if filtered_lm_eval_results:
        leaderboard_score = round(sum(filtered_lm_eval_results.values()) / len(filtered_lm_eval_results), 4)
        filtered_lm_eval_results["leaderboard"] = leaderboard_score
        print(f"  > Leaderboard Average Score: {leaderboard_score:.4f}")
        
    # --- SAVE CACHE ---
    print("Saving Phase 1 results to cache...")
    cache_data = {
        "filter_version": 2,
        "filtered": filtered_lm_eval_results,
        "raw": raw_results["results"]
    }
    with open(cache_path, "w") as f:
        json.dump(cache_data, f, indent=4)
        
    print("Phase 1 LM Eval complete.")
    return filtered_lm_eval_results, raw_results["results"]

def run_phase4_judge_evaluation(
    suite,
    model_type,
    dataset_size,
    topk,
    ref_model_seed: Optional[int] = None,
) -> Dict:
    """Phase 2: AlpacaEval evaluation with separate generation and judging stages."""
    if not suite.eval.cfg.alpacaeval.run:
        print("Judge eval disabled in suite config. Skipping Phase 4.")
        return {}

    print("\n" + "=" * 60)
    print("PHASE 2: Running AlpacaEval evaluation...")
    print("=" * 60)

    if model_type == ModelType.BASE:
        model_path = suite.sft.cfg.base_model_id
        eval_output_dir = suite.dirs.get_base_model_eval_by_seed(suite.seed)
    elif model_type == ModelType.ADAPTER:
        assert dataset_size is not None and topk is not None, (
            "dataset_size and topk must be provided for adapter judge evaluation."
        )
        model_path = suite.dirs.get_trained_model_root(
            dataset_size=cast(int, dataset_size),
            topk=cast(int, topk),
        ).as_posix()
        eval_output_dir = suite.dirs.get_trained_model_eval(
            dataset_size=cast(int, dataset_size),
            topk=cast(int, topk),
        )
    elif model_type == ModelType.REFERENCE:
        if ref_model_seed is None:
            raise ValueError("ref_model_seed must be provided for reference model judge evaluation.")
        model_path = suite.dirs.get_ref_model_root_by_seed(ref_model_seed).as_posix()
        eval_output_dir = suite.dirs.get_ref_model_eval_by_seed(ref_model_seed)
    else:
        raise ValueError(
            f"Invalid model type {model_type.value.upper()} for judge evaluation."
        )

    gc.collect()
    torch.cuda.empty_cache()

    leader_row, raw = run_alpaca_eval_from_model(
        suite=suite,
        model_path=model_path,
        eval_output_dir=eval_output_dir,
        model_type=model_type,
        dataset_size=dataset_size,
        topk=topk,
        ref_model_seed=ref_model_seed,
    )

    log_file = eval_output_dir / "alpaca_eval_raw.json"
    if not log_file.exists():
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "w") as f:
            json.dump(raw, f, indent=4)

    print(
        f"AlpacaEval complete: alpacaeval={leader_row.get('alpacaeval')}"
    )
    return leader_row
def run_phase5_save_results(
    raw_lm_eval_scores, eval_scores, suite, model_type, dataset_size, topk, save_results_and_config, ref_model_seed
) -> Optional[Dict]:
    """Phase 5: Save Configuration and Results"""
    print("\n" + "="*60)
    print("PHASE 3: Saving configuration and results...")
    print("="*60)
    
    if not save_results_and_config:
        print("save_results_and_config is False. Skipping save.")
        return None
        
    print("Writing evaluation results and configuration to disk...")
    cfg = write_eval_results_and_config(
        raw_lm_eval_scores=raw_lm_eval_scores,
        eval_scores=eval_scores,
        suite=suite,
        model_type=model_type,
        dataset_size=dataset_size,
        topk=topk,
        ref_model_seed=ref_model_seed,
    )
    print("Phase 5 complete. Results saved successfully.")
    return cfg



def eval_and_save(
    model,
    tokenizer,
    suite,
    model_type,
    gen_cache_dir: Path,
    dataset_size: Optional[int] = None,
    topk: Optional[int] = None,
    save_results_and_config: bool = True,
    ref_model_seed: Optional[int] = None,
) -> Tuple[dict, Optional[dict]]:

    print("CHANGING PADDING SIDE TO 'left' FOR INFERENCE")
    tokenizer.padding_side = 'left'

    print("\n\n" + "#"*70)
    print("### STARTING EVALUATION AND SAVE PIPELINE ###")
    print("#"*70)

    batch_size = suite.eval.params.batch_size
    eval_scores = {}

    print("Setting model to eval mode and flushing initial VRAM...")
    model.eval()
    torch.cuda.empty_cache()
    gc.collect()

    # =====================================================================
    # PHASE 1: LM Eval Evaluation
    # =====================================================================
    filtered_lm_eval_results, raw_lm_eval_scores = {}, {}
    if suite.eval.cfg.lm_eval.run:
        filtered_lm_eval_results, raw_lm_eval_scores = run_phase1_lm_eval(
            model, tokenizer, suite, gen_cache_dir
        )
        eval_scores["lm_eval_results"] = filtered_lm_eval_results
    else:
        print("LM eval disabled in suite config. Skipping Phase 1.")

    # =====================================================================
    # PHASE 2: AlpacaEval Evaluation
    # =====================================================================
    if suite.eval.cfg.alpacaeval.run:
        print(f"\nModel type is {model_type}. Running AlpacaEval evaluation.")
        if model is not None:
            remove_hook_from_module(model, recurse=True)
            model.cpu()
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        torch.cuda.empty_cache()

        eval_scores["reference_results"] = run_phase4_judge_evaluation(
            suite=suite,
            model_type=model_type,
            dataset_size=dataset_size,
            topk=topk,
            ref_model_seed=ref_model_seed,
        )
    else:
        print("AlpacaEval disabled in suite config. Skipping Phase 2.")

    # =====================================================================
    # PHASE 3: Save Configuration and Results
    # =====================================================================
    cfg = run_phase5_save_results(
        raw_lm_eval_scores,
        eval_scores,
        suite,
        model_type,
        dataset_size,
        topk,
        save_results_and_config,
        ref_model_seed,
    )
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()

    print("\n" + "#"*70)
    print("### EVALUATION AND SAVE PIPELINE FINISHED SUCCESSFULLY ###")
    print("#"*70 + "\n")

    return eval_scores, cfg
