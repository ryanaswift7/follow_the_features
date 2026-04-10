import argparse
import gc
import json

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.trainer.sft_trainer import SFTTrainer

from src.prune import apply_chat_formatting_with_tokenizer_alpaca
from src.schemas.suite import SuiteConfig
from src.sft import PEFT_CONFIG, format_to_conversation, generate_sft_trainer_args
from src.utils.constants import EVAL_CONFIG_FILENAME, EVAL_RESULTS_FILENAME
from src.utils.enums import ModelType
from src.utils.evals import eval_and_save


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate null distribution for experiment suite based on configuration file."
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Path to the experiment configuration YAML file.",
    )
    parser.add_argument(
        "--num-ref-models",
        "-n",
        type=int,
        default=3,
        help="Number of reference models to generate (default: 3).",
    )
    return parser.parse_args()


def generate_reference_models(suite: SuiteConfig):
    num_ref_models = suite.eval.cfg.num_ref_models
    dataset_id = suite.prune.cfg.dataset_id
    batch_size = suite.prune.params.batch_size
    seed = suite.seed
    base_model_id = suite.sft.cfg.base_model_id

    # Define overall output paths
    overall_results_path = suite.dirs.reference_model_eval / EVAL_RESULTS_FILENAME
    overall_config_path = suite.dirs.reference_model_eval / EVAL_CONFIG_FILENAME

    print(f"\n{'=' * 50}")
    print("[REF MODELS] Initializing Reference Model Generator")
    print(f"  - Dataset:      {dataset_id}")
    print(f"  - Base Model:   {base_model_id}")
    print(f"  - Ref Models:   {num_ref_models}")
    print("  - Training on:  FULL DATASET")
    print(f"{'=' * 50}\n")

    # 1. OVERALL CHECK: Skip everything if the final aggregated files already exist
    if overall_results_path.exists() and overall_config_path.exists():
        print(
            "[REF MODELS] Overall evaluation results and config already present. Skipping generation."
        )
        with open(overall_results_path, "r") as f:
            final_mean_scores = json.load(f)
        with open(overall_config_path, "r") as f:
            cfg = json.load(f)
        return final_mean_scores, cfg

    # 2. PRE-CHECK INDIVIDUAL MODELS: Determine if we actually need to load the dataset
    models_to_train = []
    for i in range(num_ref_models):
        trial_seed = seed + i
        trial_output_dir = suite.dirs.get_ref_model_root_by_seed(trial_seed)
        if not (
            (trial_output_dir / EVAL_RESULTS_FILENAME).exists()
            and (trial_output_dir / EVAL_CONFIG_FILENAME).exists()
        ):
            models_to_train.append(i)

    # 3. CONDITIONALLY LOAD DATA: Only load tokenizer and map datasets if at least one model needs it
    if models_to_train:
        print(f"[REF MODELS] Loading tokenizer: {suite.sft.cfg.tokenizer_id}...")
        tokenizer = AutoTokenizer.from_pretrained(suite.sft.cfg.tokenizer_id)

        print(f"[REF MODELS] Loading dataset: {dataset_id}...")
        full_ds = load_dataset(dataset_id, split="train")

        if suite.debug_mode:
            full_ds = full_ds.select(range(1))

        full_ds_len = len(full_ds)

        print(f"[REF MODELS] Applying chat formatting (Batch Size: {batch_size})...")
        full_ds = full_ds.map(
            lambda examples: apply_chat_formatting_with_tokenizer_alpaca(tokenizer, examples),
            batched=True,
            batch_size=batch_size,
            desc="Tokenizing Data",
        )
        full_ds = full_ds.map(
            format_to_conversation,
            remove_columns=full_ds.column_names,
        )
    else:
        print(
            "[REF MODELS] All individual reference models have existing results. Skipping dataset loading."
        )

    all_trials_scores = []
    cfg = None  # Fallback config

    print("\n[REF MODELS] Processing Reference Models...")
    pbar_models = tqdm(
        range(num_ref_models), desc="Training/Loading Ref Models", position=0
    )

    for i in pbar_models:
        trial_seed = seed + i
        trial_output_dir = suite.dirs.get_ref_model_root_by_seed(trial_seed)
        trial_output_dir.mkdir(parents=True, exist_ok=True)

        trial_results_path = trial_output_dir / EVAL_RESULTS_FILENAME
        trial_config_path = trial_output_dir / EVAL_CONFIG_FILENAME

        pbar_models.set_description(f"Ref Model {i + 1}/{num_ref_models}")

        # 4. INDIVIDUAL TRIAL CHECK: Load results if they exist, skip everything else
        if trial_results_path.exists() and trial_config_path.exists():
            print(
                f"\n[REF MODELS] Results for Ref Model {i + 1} already present. Loading cached results."
            )
            with open(trial_results_path, "r") as f:
                current_scores = json.load(f)
            with open(trial_config_path, "r") as f:
                cfg = json.load(f)

            all_trials_scores.append(current_scores)
            continue

        # 5. MODEL WEIGHTS CHECK: If model is saved but no results exist, load model and skip training
        has_full_model = (trial_output_dir / "config.json").exists()
        has_adapter = (trial_output_dir / "adapter_config.json").exists()

        if has_full_model or has_adapter:
            print(
                f"\n[REF MODELS] Model weights found in {trial_output_dir}. Skipping training, loading for evaluation."
            )

            if has_adapter:
                # Load base model then apply PEFT adapter
                eval_model = AutoModelForCausalLM.from_pretrained(
                    base_model_id, device_map="auto", torch_dtype=torch.bfloat16
                )
                eval_model.load_adapter(str(trial_output_dir))
            else:
                # Load full finetuned model
                eval_model = AutoModelForCausalLM.from_pretrained(
                    trial_output_dir, device_map="auto", torch_dtype=torch.bfloat16
                )
        else:
            # --- Training Execution (if model doesn't exist) ---
            dataset = full_ds.shuffle(seed=trial_seed)

            eval_model = AutoModelForCausalLM.from_pretrained(
                base_model_id, device_map="auto", torch_dtype=torch.bfloat16
            )

            trainer_args = generate_sft_trainer_args(
                suite=suite, output_dir=str(trial_output_dir)
            )
            trainer_args.disable_tqdm = True

            trainer = SFTTrainer(
                model=eval_model,
                args=trainer_args,
                train_dataset=dataset,
                peft_config=PEFT_CONFIG,
                processing_class=tokenizer,
            )

            trainer.train()
            merged_model = trainer.model.merge_and_unload()
            merged_model.save_pretrained(trial_output_dir)
            tokenizer.save_pretrained(trial_output_dir)

            eval_model = merged_model

        # --- Evaluation Execution ---
        current_scores, cfg = eval_and_save(
            model=eval_model,
            tokenizer=tokenizer,
            suite=suite,
            model_type=ModelType.REFERENCE,
            dataset_size=full_ds_len,
            topk=None,
            save_results_and_config=True,
            ref_model_seed=trial_seed,
        )

        all_trials_scores.append(current_scores["lm_eval_results"])

        # Cleanup memory
        del eval_model
        if "trainer" in locals():
            del trainer
        gc.collect()
        torch.cuda.empty_cache()

    print("\n[REF MODELS] Calculating Mean Scores across Reference Models...")

    final_mean_scores = {}

    if all_trials_scores:
        keys = all_trials_scores[0].keys()
        for key in keys:
            values = [trial.get(key) for trial in all_trials_scores if key in trial]
            final_mean_scores[key] = round(sum(values) / len(values), 4)
    
    final_results = {"lm_eval_results": final_mean_scores}

    # OVERALL SAVE: Save the aggregated results so we can skip this entirely next time
    suite.dirs.reference_model_eval.mkdir(parents=True, exist_ok=True)
    with open(overall_results_path, "w") as f:
        json.dump(final_results, f, indent=4)
    with open(overall_config_path, "w") as f:
        json.dump(cfg, f, indent=4)

    print("[REF MODELS] Generation Complete.")
    return final_results, cfg


def main():
    args = parse_args()
    config_path = args.config

    suite = SuiteConfig.from_yaml(config_path)
    generate_reference_models(suite=suite)


if __name__ == "__main__":
    main()
