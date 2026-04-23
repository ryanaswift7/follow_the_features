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
from src.utils.evaluator import Evaluator
from src.utils.ollama_manager import OllamaManager


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


def run_phase1_lm_eval(model, tokenizer, suite) -> Tuple[Dict, Dict]:
    """Phase 1: LM Eval Evaluation"""
    print("\n" + "="*60)
    print("PHASE 1: Starting LM Eval Evaluation...")
    print("="*60)
    
    lm_eval_results = {}
    raw_lm_eval_scores = {}
    
    if suite.eval.cfg.tasks is None:
        print("No LM Eval tasks specified in suite. Skipping Phase 1.")
        return {}, {}

    print("Preparing model for LM Eval: Saving device map and moving to cuda:0...")
    original_map = model.hf_device_map
    remove_hook_from_module(model, recurse=True)
    model.to("cuda:0")
    torch.cuda.empty_cache()
    gc.collect()

    print("Wrapping existing model in HFLM for lm_eval (reusing loaded VRAM)...")
    hflm_model = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=suite.eval.params.batch_size,
    )

    print(f"Running simple_evaluate on tasks: {suite.eval.cfg.tasks}...")
    raw_results = lm_eval.simple_evaluate(
        model=hflm_model, 
        tasks=suite.eval.cfg.tasks,
        limit=suite.eval.cfg.task_limit,
        batch_size=suite.eval.params.batch_size,
        gen_kwargs={
            "max_gen_toks": suite.eval.cfg.max_new_tokens,
        },
        apply_chat_template=True,
    )

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
        
    print("Phase 1 LM Eval complete.")
    return filtered_lm_eval_results, raw_results["results"]


def run_phase2_generate_candidates(model, tokenizer, suite, datasets_map, batch_size) -> Dict:
    """Phase 2: Generate Candidate Responses"""
    print("\n" + "="*60)
    print("PHASE 2: Generating Candidate Responses...")
    print("="*60)
    
    candidate_data = {}
    for ds_name, data in datasets_map.items():
        print(f"Processing dataset '{ds_name}' ({len(data)} samples)...")
        candidate_preds, metric_refs = generate_responses(
            model,
            tokenizer,
            data,
            batch_size=batch_size,
            max_new_tokens=suite.eval.cfg.max_new_tokens,
        )
        candidate_data[ds_name] = {
            "preds": candidate_preds, 
            "refs": metric_refs, 
            "data": data
        }
        print(f"Successfully generated candidates for '{ds_name}'. Cleaning up VRAM...")
        torch.cuda.empty_cache()
        gc.collect()
        
    print("Phase 2 complete. All candidate responses generated.")
    return candidate_data


def run_phase3_generate_references(suite, tokenizer, datasets_map, batch_size) -> Dict:
    """Phase 3: Generate Reference Responses (Inverted Loop to save RAM)"""
    print("\n" + "="*60)
    print(f"PHASE 3: Generating Reference Responses ({suite.eval.cfg.num_ref_models} models)...")
    print("="*60)
    
    ref_outputs_by_dataset = {ds_name: [] for ds_name in datasets_map.keys()}

    for i in range(suite.eval.cfg.num_ref_models):
        torch.cuda.empty_cache()
        gc.collect()
        
        ref_path = suite.dirs.get_ref_model_root_by_seed(suite.seed + i)
        print(f"\n[{i+1}/{suite.eval.cfg.num_ref_models}] Loading reference model weights into VRAM from: {ref_path}")

        ref_model = AutoModelForCausalLM.from_pretrained(
            ref_path, device_map="auto", torch_dtype=torch.bfloat16
        )
        # ref_tokenizer = AutoTokenizer.from_pretrained(ref_path, fix_mistral_regex=True)

        for ds_name, data in datasets_map.items():
            print(f"  > Generating baselines for dataset '{ds_name}'...")
            ref_preds, _ = generate_responses(
                ref_model,
                tokenizer,
                data,
                batch_size=batch_size,
                max_new_tokens=suite.eval.cfg.max_new_tokens,
            )
            ref_outputs_by_dataset[ds_name].append(ref_preds)
            torch.cuda.empty_cache()
            gc.collect()

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
    dataset_size: Optional[int] = None,
    topk: Optional[int] = None,
    save_results_and_config: bool = True,
    ref_model_seed: Optional[int] = None,
) -> Tuple[dict, Optional[dict]]:

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
    filtered_lm_eval_results, raw_lm_eval_scores = run_phase1_lm_eval(model, tokenizer, suite)
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
            model, tokenizer, suite, datasets_map, batch_size
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
        print("\nModel type is REFERENCE. Skipping Phases 2, 3, and 4 (Judge Evaporation).")

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
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_new_tokens,  # model.config.max_position_embeddings, # // 4
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                # do_sample=False,  # Greedy decoding for deterministic eval
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode and strip the prompt
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for j, text in enumerate(decoded):
            # Heuristic to strip input prompt if it leaks into output
            # (Gemma chat template usually handles this, but we ensure cleanliness)
            response = text.split("model")[-1] if "model" in text else text
            predictions.append(response.strip())
        del inputs
        del outputs
        torch.cuda.empty_cache()

    return predictions, refs
