import gc
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, cast

import lm_eval
from lm_eval.models.huggingface import HFLM
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate.hooks import remove_hook_from_module
from accelerate import dispatch_model
from datetime import datetime

from src.schemas.suite import SuiteConfig
from src.utils.constants import (
    EVAL_CONFIG_FILENAME,
    EVAL_RESULTS_FILENAME,
    INVALID_JUDGE_LOG_FILENAME,
)
from src.utils.enums import ModelType
from src.custom_judge.evaluator import Evaluator
from src.custom_judge.ollama_manager import OllamaManager


def write_eval_results_and_config(
    raw_lm_eval_scores: Dict,
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
    """Phase 1: LM Eval Evaluation with full caching"""
    print("\n" + "="*60)
    print("PHASE 1: Starting LM Eval Evaluation...")
    print("="*60)
    
    if suite.eval.cfg.tasks is None:
        print("No LM Eval tasks specified in suite. Skipping Phase 1.")
        return {}, {}

    # Ensure the cache directory exists and define the cache path
    gen_cache_dir = Path(gen_cache_dir)
    gen_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = gen_cache_dir / "phase1_lm_eval.json"

    # --- CACHE CHECK ---
    if cache_path.exists():
        print(f"Found cached Phase 1 results at {cache_path}. Skipping evaluation.")
        with open(cache_path, "r") as f:
            cache_data = json.load(f)
        return cache_data["filtered"], cache_data["raw"]

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

    print(f"Running simple_evaluate on tasks: {suite.eval.cfg.tasks}...")


    # from transformers import GenerationConfig

    # inspect what the model is actually carrying
    hflm_model.model.generation_config.max_new_tokens = suite.eval.cfg.max_new_tokens
    # print(hflm_model.model.generation_config)

    # force a clean generation config
    # hflm_model.model.generation_config = GenerationConfig(max_new_tokens=256)

    raw_results = lm_eval.simple_evaluate(
        model=hflm_model, 
        tasks=suite.eval.cfg.tasks,
        limit=suite.eval.cfg.task_limit,
        batch_size=suite.eval.params.batch_size,
        gen_kwargs={"max_gen_toks": suite.eval.cfg.max_new_tokens},
        # if "olmo" not in suite.sft.cfg.base_model_id.lower() else 
        #     {"max_new_tokens": suite.eval.cfg.max_new_tokens,},
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

    selected_keys = [
        "leaderboard_ifeval",
        "leaderboard_bbh",
        "leaderboard_math_hard",
        "leaderboard_gpqa",
        "leaderboard_musr",
        "leaderboard_mmlu_pro",
    ]
    
    filtered_lm_eval_results = {k: v for k, v in lm_eval_results.items() if k in selected_keys}
    
    if filtered_lm_eval_results:
        avg_score = round(sum(list(filtered_lm_eval_results.values())) / len(filtered_lm_eval_results), 4)
        filtered_lm_eval_results["average_lm_eval_score"] = avg_score
        print(f"  > Filtered Average Score : {avg_score:.4f}")
        
    # --- SAVE CACHE ---
    print("Saving Phase 1 results to cache...")
    cache_data = {
        "filtered": filtered_lm_eval_results,
        "raw": raw_results["results"]
    }
    with open(cache_path, "w") as f:
        json.dump(cache_data, f, indent=4)
        
    print("Phase 1 LM Eval complete.")
    return filtered_lm_eval_results, raw_results["results"]


def run_phase2_generate_candidates(model, tokenizer, suite, datasets_map, batch_size, candidate_gen_cache_dir: Path) -> Dict:
    """Phase 2: Generate Candidate Responses with chunked dataset-level caching"""
    print("\n" + "="*60)
    print("PHASE 2: Generating Candidate Responses...")
    print("="*60)
    
    CHUNK_SIZE = 10
    candidate_data = {}
    
    # Ensure the cache directory exists
    candidate_gen_cache_dir = Path(candidate_gen_cache_dir)
    candidate_gen_cache_dir.mkdir(parents=True, exist_ok=True)
    
    for ds_name, data in datasets_map.items():
        cache_path = candidate_gen_cache_dir / f"{ds_name}.json"
        total_samples = len(data)
        
        if cache_path.exists():
            with open(cache_path, 'r') as f:
                cache = json.load(f)
        else:
            cache = {"preds": [], "refs": []}
            
        num_cached = len(cache["preds"])
        
        # Handle over-cached data
        if num_cached > total_samples:
            print(f"Cache for '{ds_name}' has {num_cached} samples. Truncating to requested {total_samples}.")
            cache["preds"] = cache["preds"][:total_samples]
            if cache.get("refs"):
                cache["refs"] = cache["refs"][:total_samples]
            num_cached = total_samples
        
        if num_cached < total_samples:
            print(f"Processing '{ds_name}': Resuming from sample {num_cached}/{total_samples}...")
            
            for j in range(num_cached, total_samples, CHUNK_SIZE):
                end_idx = min(j + CHUNK_SIZE, total_samples)
                
                # Use HF .select() to preserve Dataset object type
                # chunk = data.select(range(j, end_idx))
                chunk = data[j:end_idx]
                
                new_preds, new_refs = generate_responses(
                    model,
                    tokenizer,
                    chunk,
                    batch_size=batch_size,
                    max_new_tokens=suite.eval.cfg.max_new_tokens,
                )
                
                cache["preds"].extend(new_preds)
                if new_refs is not None:
                    cache["refs"].extend(new_refs)
                    
                with open(cache_path, 'w') as f:
                    json.dump(cache, f)
                    
                print(f"  > Saved chunk: {end_idx}/{total_samples} generations cached.")
                
                torch.cuda.empty_cache()
                gc.collect()
        else:
            print(f"Dataset '{ds_name}' fully cached ({num_cached}/{total_samples}). Skipping generation.")
            
        candidate_data[ds_name] = {
            "preds": cache["preds"], 
            "refs": cache["refs"], 
            "data": data
        }
        
    print("Phase 2 complete. All candidate responses generated.")
    return candidate_data


def run_phase3_generate_references(suite, tokenizer, datasets_map, batch_size) -> Dict:
    """Phase 3: Generate Reference Responses with VRAM optimization and chunked caching"""
    print("\n" + "="*60)
    print(f"PHASE 3: Generating Reference Responses ({suite.eval.cfg.num_ref_models} models)...")
    print("="*60)
    
    CHUNK_SIZE = 10
    ref_outputs_by_dataset = {ds_name: [] for ds_name in datasets_map.keys()}

    for i in range(suite.eval.cfg.num_ref_models):
        ref_path = suite.dirs.get_ref_model_root_by_seed(suite.seed + i)
        ref_gen_cache_dir = Path(suite.dirs.get_ref_model_gen_cache_by_seed(
            seed=suite.seed + i, max_new_tokens=suite.eval.cfg.max_new_tokens
        ))
        
        # Ensure the cache directory exists for this specific reference model
        ref_gen_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # --- PRE-CHECK: Are ALL datasets fully cached for this model? ---
        all_fully_cached = True
        for ds_name, data in datasets_map.items():
            cache_path = ref_gen_cache_dir / f"{ds_name}.json"
            if not cache_path.exists():
                all_fully_cached = False
                break
            with open(cache_path, 'r') as f:
                if len(json.load(f)["preds"]) < len(data):
                    all_fully_cached = False
                    break
        
        if all_fully_cached:
            print(f"\n[{i+1}/{suite.eval.cfg.num_ref_models}] All datasets fully cached for {ref_path}. SKIPPING VRAM LOAD.")
            for ds_name, data in datasets_map.items():
                cache_path = ref_gen_cache_dir / f"{ds_name}.json"
                with open(cache_path, 'r') as f:
                    preds = json.load(f)["preds"]
                    # Slice to exact dataset length in case of over-caching
                    ref_outputs_by_dataset[ds_name].append(preds[:len(data)])
            continue

        # If we reach here, we have pending generations and need the model in VRAM
        torch.cuda.empty_cache()
        gc.collect()
        print(f"\n[{i+1}/{suite.eval.cfg.num_ref_models}] Loading reference model weights into VRAM from: {ref_path}")

        ref_model = AutoModelForCausalLM.from_pretrained(
            ref_path, device_map="auto", torch_dtype=torch.bfloat16
        )

        for ds_name, data in datasets_map.items():
            cache_path = ref_gen_cache_dir / f"{ds_name}.json"
            total_samples = len(data)
            
            if cache_path.exists():
                with open(cache_path, 'r') as f:
                    cache = json.load(f)
            else:
                cache = {"preds": []}
                
            num_cached = len(cache["preds"])
            
            if num_cached > total_samples:
                print(f"  > Cache has {num_cached} samples. Truncating to requested {total_samples}.")
                cache["preds"] = cache["preds"][:total_samples]
                num_cached = total_samples
            
            if num_cached < total_samples:
                print(f"  > Generating baselines for '{ds_name}': Resuming from sample {num_cached}/{total_samples}...")
                for j in range(num_cached, total_samples, CHUNK_SIZE):
                    end_idx = min(j + CHUNK_SIZE, total_samples)
                    
                    # Use HF .select() to preserve Dataset object type
                    # chunk = data.select(range(j, end_idx))
                    chunk = data[j:end_idx]
                    
                    new_preds, _ = generate_responses(
                        ref_model,
                        tokenizer,
                        chunk,
                        batch_size=batch_size,
                        max_new_tokens=suite.eval.cfg.max_new_tokens,
                    )
                    
                    cache["preds"].extend(new_preds)
                    with open(cache_path, 'w') as f:
                        json.dump(cache, f)
                        
                    print(f"    > Saved chunk: {end_idx}/{total_samples} generations cached.")
                    
                    torch.cuda.empty_cache()
                    gc.collect()
            else:
                 print(f"  > Dataset '{ds_name}' fully cached.")
            
            ref_outputs_by_dataset[ds_name].append(cache["preds"])

        print(f"[{i+1}/{suite.eval.cfg.num_ref_models}] Complete. Deleting reference model from VRAM to make room...")
        remove_hook_from_module(ref_model, recurse=True)
        ref_model.cpu()
        del ref_model
        gc.collect()
        torch.cuda.empty_cache()
        
    print("Phase 3 complete. All reference responses generated.")
    return ref_outputs_by_dataset


def run_phase4_judge_evaluation(
    suite, model_type, datasets_map, candidate_data, ref_outputs_by_dataset, dataset_size, topk
) -> Dict:
    """Phase 4: Construct Payload and Evaluate with Judge"""
    # Final safety flush before loading the Judge
    gc.collect()
    torch.cuda.empty_cache()

    print("\n" + "="*60)
    print("PHASE 4: Constructing Payload and Evaluating with Judge...")
    print("="*60)
    
    model_scores = {}
    
    print("Initializing OllamaManager and Evaluator...")
    with OllamaManager(ollama_judge=suite.eval.cfg.ollama_judge) as ol_man:
        evaluator = Evaluator(ollama_judge=suite.eval.cfg.ollama_judge)

        for ds_name in datasets_map.keys():
            print(f"\nBuilding evaluation payload for dataset: '{ds_name}'...")
            eval_payload = []
            data = candidate_data[ds_name]["data"]
            cand_preds = candidate_data[ds_name]["preds"]
            metric_refs = candidate_data[ds_name]["refs"]
            ref_model_outputs = ref_outputs_by_dataset[ds_name]

            for i in range(len(cand_preds)):
                refs_for_this_sample = [run[i] for run in ref_model_outputs]
                eval_payload.append(
                    {
                        "prompt": data[i]["prompt"],
                        "generation": cand_preds[i],
                        "reference_metrics": metric_refs[i],
                        "reference_judges": refs_for_this_sample,
                    }
                )

            print("Determining log file path based on model type...")
            log_filename = f"{ds_name}_{INVALID_JUDGE_LOG_FILENAME}"
            if model_type == ModelType.BASE:
                log_file = suite.dirs.base_model_eval / log_filename
            elif model_type == ModelType.ADAPTER:
                log_file = (
                    suite.dirs.get_trained_model_eval(
                        dataset_size=cast(int, dataset_size), topk=cast(int, topk)
                    )
                    / log_filename
                )
            elif model_type == ModelType.SOTA:
                log_file = Path(f"sota_eval_logs/{datetime.now().strftime('%Y%m%d_%H%M%S')}/{log_filename}")
            else:
                raise ValueError(f"Invalid model type {model_type.value.upper()} for judge evaluation.")
            
            log_file.parent.mkdir(parents=True, exist_ok=True)
            print(f"Log file set to: {log_file}")
            
            print(f"Sending {len(eval_payload)} samples to LLM judge for '{ds_name}'...")
            dataset_means = evaluator.run_full_eval(
                test_data=eval_payload, port=ol_man.port, log_file=log_file
            )

            judge = round(dataset_means["judge_score"], 3)
            invalid = dataset_means["invalid_judge_responses"]
            
            print(f"Scores for '{ds_name}': Judge={judge}, Invalid={invalid}")

            model_scores[ds_name] = {
                "judge_score": judge,
                "invalid_judge_responses": invalid,
            }

    print("\nAggregating final reference results...")
    if model_scores:
        avg_judge = sum(m["judge_score"] for m in model_scores.values()) / len(model_scores)
        print(f"Average Judge Score: {avg_judge:.3f}")
        return {
            "average_judge_score": round(avg_judge, 3),
            **model_scores,
        }
    
    print("No model scores computed. Returning zero averages.")
    return {"average_judge_score": 0}


def run_phase5_save_results(
    raw_lm_eval_scores, eval_scores, suite, model_type, dataset_size, topk, save_results_and_config, ref_model_seed
) -> Optional[Dict]:
    """Phase 5: Save Configuration and Results"""
    print("\n" + "="*60)
    print("PHASE 5: Saving Configuration and Results...")
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
    # model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bloat16, )
    model.eval()
    torch.cuda.empty_cache()
    gc.collect()

    # =====================================================================
    # PHASE 1: LM Eval Evaluation
    # =====================================================================
    filtered_lm_eval_results, raw_lm_eval_scores = run_phase1_lm_eval(model, tokenizer, suite, gen_cache_dir)
    eval_scores["lm_eval_results"] = filtered_lm_eval_results


    # =====================================================================
    # PHASE 2-4: Candidate Gen, Baseline Gen, and Judge Eval
    # =====================================================================
    if model_type != ModelType.REFERENCE:
        print(f"\nModel type is {model_type}. Proceeding to response generation and judge eval.")
        print("Loading evaluation datasets into memory...")
        datasets_map = load_eval_datasets(
            seed=suite.seed, limit=suite.eval.cfg.num_samples
        )
        
        # PHASE 2
        candidate_data = run_phase2_generate_candidates(
            model, tokenizer, suite, datasets_map, batch_size, candidate_gen_cache_dir=gen_cache_dir,
        )

        # Explicitly clean up candidate model/tokenizer to free VRAM for references
        print("\n" + "-"*60)
        print("Candidate generation finished. Deleting candidate model from VRAM to make room for reference models...")
        print("-"*60)
        remove_hook_from_module(model, recurse=True)
        model.cpu()
        del model
        gc.collect()
        torch.cuda.empty_cache()

        # PHASE 3
        ref_outputs_by_dataset = run_phase3_generate_references(
            suite, tokenizer, datasets_map, batch_size
        )

        # PHASE 4
        eval_scores["reference_results"] = run_phase4_judge_evaluation(
            suite, model_type, datasets_map, candidate_data, 
            ref_outputs_by_dataset, dataset_size, topk
        )
    else:
        print("\nModel type is REFERENCE. Skipping Phases 2, 3, and 4 (Judge Evaluation).")

    # =====================================================================
    # PHASE 5: Save Configuration and Results
    # =====================================================================
    cfg = run_phase5_save_results(
        raw_lm_eval_scores, eval_scores, suite, model_type, 
        dataset_size, topk, save_results_and_config, ref_model_seed
    )
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()

    print("\n" + "#"*70)
    print("### EVALUATION AND SAVE PIPELINE FINISHED SUCCESSFULLY ###")
    print("#"*70 + "\n")

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
    ds = load_dataset("ryanaswift7/natural-instructions-10k", split="v1")
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

    # CRITICAL: Generative models must use left-padding
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    prompts = [x["prompt"] for x in dataset]
    refs = [x["reference"] for x in dataset]

    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch_prompts = prompts[i : i + batch_size]

        # Apply Chat Template
        formatted_prompts = []
        for p in batch_prompts:
            msgs = [{"role": "user", "content": p}]
            formatted_prompts.append(
                tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
            )

        # Tokenize (removed max_new_tokens truncation bug)
        inputs = tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_new_tokens, # model.config.max_position_embeddings, # // 4
        ).to(model.device)


        # Get the sequence length of the input prompts
        input_length = inputs.input_ids.shape[1]

        # Use inference_mode instead of no_grad for a slight speedup
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                # do_sample=False, # Uncomment if strict greedy decoding is desired
            )

        # SLICE the output tensor to isolate only the newly generated tokens
        generated_tokens = outputs[:, input_length:]

        # Decode only the generated response
        decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        predictions.extend([text.strip() for text in decoded])
        
        # Free up memory references, but don't force an expensive CUDA cache flush
        del inputs
        del outputs
        del generated_tokens

    return predictions, refs
