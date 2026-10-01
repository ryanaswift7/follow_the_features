import argparse
import gc
import json
from typing import Any, Dict

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from trl.trainer.sft_trainer import SFTTrainer

from src.prune import get_formatting_function
from src.schemas.suite import SuiteConfig
from src.sft import PEFT_CONFIG, generate_sft_trainer_args
from src.utils.constants import EVAL_CONFIG_FILENAME, EVAL_RESULTS_FILENAME
from src.utils.enums import ModelType
from src.utils.evals import eval_and_save
from src.utils.load_model_and_tokenizer import load_model_and_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate reference models for an experiment suite based on a configuration file."
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


def _mean_numeric_dicts(dicts: list[dict[str, Any]]) -> dict[str, Any]:
    if not dicts:
        return {}

    numeric_keys = set()
    for item in dicts:
        for key, value in item.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric_keys.add(key)

    aggregated: dict[str, Any] = {}
    for key in sorted(numeric_keys):
        values = [
            item[key]
            for item in dicts
            if isinstance(item.get(key), (int, float)) and not isinstance(item.get(key), bool)
        ]
        if values:
            aggregated[key] = round(sum(values) / len(values), 4)
    return aggregated


def _trial_has_valid_results(trial_output_dir, suite: SuiteConfig, expected_sections: list[str]) -> bool:
    results_path = trial_output_dir / EVAL_RESULTS_FILENAME
    config_path = trial_output_dir / EVAL_CONFIG_FILENAME
    if not (results_path.exists() and config_path.exists()):
        return False

    try:
        with open(results_path, "r") as f:
            results = json.load(f)
        with open(config_path, "r") as f:
            cfg = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False

    if cfg.get("eval_id") != suite.ids.eval:
        return False

    return all(isinstance(results, dict) and results.get(section) is not None for section in expected_sections)


def generate_reference_models(
    suite: SuiteConfig,
    run_lm_eval: bool = True,
    run_judge_eval: bool = True,
):
    num_ref_models = suite.eval.cfg.num_ref_models
    dataset_id = suite.prune.cfg.dataset_id
    batch_size = suite.prune.params.batch_size
    seed = suite.seed
    base_model_id = suite.sft.cfg.base_model_id

    overall_results_path = suite.dirs.reference_model_summary_eval / EVAL_RESULTS_FILENAME
    overall_config_path = suite.dirs.reference_model_summary_eval / EVAL_CONFIG_FILENAME

    print(f"\n{'=' * 50}")
    print("[REF MODELS] Initializing Reference Model Generator")
    print(f"  - Dataset:      {dataset_id}")
    print(f"  - Base Model:   {base_model_id}")
    print(f"  - Ref Models:   {num_ref_models}")
    print("  - Training on:  FULL DATASET")
    print(f"{'=' * 50}\n")

    if not (run_lm_eval or run_judge_eval):
        print("[REF MODELS] Both LM eval and judge eval are disabled. Nothing to do.")
        return {}, None

    expected_sections: list[str] = []
    if run_lm_eval:
        expected_sections.append("lm_eval_results")
    if run_judge_eval:
        expected_sections.append("reference_results")

    if overall_results_path.exists() and overall_config_path.exists():
        try:
            with open(overall_results_path, "r") as f:
                final_results = json.load(f)
            with open(overall_config_path, "r") as f:
                cfg = json.load(f)
            if cfg.get("eval_id") == suite.ids.eval and all(
                isinstance(final_results, dict) and final_results.get(section) is not None
                for section in expected_sections
            ):
                print("[REF MODELS] Cached aggregate reference results found. Skipping generation.")
                return final_results, cfg
        except (json.JSONDecodeError, OSError):
            pass

    models_to_train = []
    for i in range(num_ref_models):
        trial_seed = seed + i
        trial_output_dir = suite.dirs.get_ref_model_root_by_seed(trial_seed)
        if not _trial_has_valid_results(trial_output_dir, suite, expected_sections):
            models_to_train.append(i)

    if models_to_train:
        print(f"[REF MODELS] Loading dataset: {dataset_id}...")
        if "ultrachat" in dataset_id:
            raw_full_ds = load_dataset(dataset_id, split="train_sft")
        else:
            raw_full_ds = load_dataset(dataset_id, split="train")

        if suite.debug_mode:
            raw_full_ds = raw_full_ds.select(range(1))

        if suite.prune.cfg.num_samples is not None and suite.prune.cfg.num_samples < len(raw_full_ds):
            raw_full_ds = raw_full_ds.select(range(suite.prune.cfg.num_samples))

        full_ds_len = len(raw_full_ds)
        if full_ds_len == 0:
            raise ValueError(
                f"Reference model training dataset is empty after applying debug/num_samples filters for {dataset_id}."
            )
    else:
        print("[REF MODELS] All individual reference models already have valid cached results.")
        raw_full_ds = None
        full_ds_len = None

    all_lm_scores: list[dict[str, Any]] = []
    all_judge_scores: list[dict[str, Any]] = []
    cfg = None

    print("\n[REF MODELS] Processing Reference Models...")
    pbar_models = tqdm(range(num_ref_models), desc="Training/Loading Ref Models", position=0)

    for i in pbar_models:
        trial_seed = seed + i
        trial_output_dir = suite.dirs.get_ref_model_root_by_seed(trial_seed)
        trial_output_dir.mkdir(parents=True, exist_ok=True)

        trial_results_path = trial_output_dir / EVAL_RESULTS_FILENAME
        trial_config_path = trial_output_dir / EVAL_CONFIG_FILENAME

        pbar_models.set_description(f"Ref Model {i + 1}/{num_ref_models}")

        if _trial_has_valid_results(trial_output_dir, suite, expected_sections):
            print(f"\n[REF MODELS] Cached results for Ref Model {i + 1} are valid. Loading.")
            with open(trial_results_path, "r") as f:
                current_scores = json.load(f)
            with open(trial_config_path, "r") as f:
                cfg = json.load(f)

            if run_lm_eval and current_scores.get("lm_eval_results") is not None:
                all_lm_scores.append(current_scores["lm_eval_results"])
            if run_judge_eval and current_scores.get("reference_results") is not None:
                all_judge_scores.append(current_scores["reference_results"])
            continue

        has_full_model = (trial_output_dir / "config.json").exists()
        has_adapter = (trial_output_dir / "adapter_config.json").exists()

        if has_full_model or has_adapter:
            print(
                f"\n[REF MODELS] Model weights found in {trial_output_dir}. Skipping training, loading for evaluation."
            )
            if has_adapter:
                eval_model, tokenizer = load_model_and_tokenizer(
                    model_id=base_model_id,
                    tokenizer_id=trial_output_dir,
                    is_inference=True,
                )
                eval_model.load_adapter(str(trial_output_dir))
            else:
                eval_model, tokenizer = load_model_and_tokenizer(
                    model_id=trial_output_dir,
                    tokenizer_id=trial_output_dir,
                    is_inference=True,
                )
        else:
            eval_model, tokenizer = load_model_and_tokenizer(
                model_id=base_model_id,
                tokenizer_id=suite.sft.cfg.tokenizer_id,
                is_inference=False,
            )

            print(f"[REF MODELS] Applying chat formatting (Batch Size: {batch_size})...")
            formatter_func = get_formatting_function(dataset_id)
            dataset = raw_full_ds.map(
                lambda examples: formatter_func(tokenizer, examples, is_inference=False),
                batched=True,
                batch_size=batch_size,
                desc="Tokenizing Data",
                remove_columns=raw_full_ds.column_names,
            )
            dataset = dataset.shuffle(seed=trial_seed)

            trainer_args = generate_sft_trainer_args(
                suite=suite,
                output_dir=str(trial_output_dir),
            )

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

        current_scores, cfg = eval_and_save(
            model=eval_model,
            tokenizer=tokenizer,
            suite=suite,
            model_type=ModelType.REFERENCE,
            dataset_size=full_ds_len,
            topk=None,
            save_results_and_config=True,
            ref_model_seed=trial_seed,
            gen_cache_dir=suite.dirs.get_ref_model_gen_cache_by_seed(
                seed=trial_seed,
                max_new_tokens=suite.eval.cfg.lm_eval.max_new_tokens,
            ),
        )

        if run_lm_eval and current_scores.get("lm_eval_results") is not None:
            all_lm_scores.append(current_scores["lm_eval_results"])
        if run_judge_eval and current_scores.get("reference_results") is not None:
            all_judge_scores.append(current_scores["reference_results"])

        del eval_model
        if "trainer" in locals():
            del trainer
        gc.collect()
        torch.cuda.empty_cache()

    final_results: dict[str, Any] = {}
    if run_lm_eval:
        final_results["lm_eval_results"] = _mean_numeric_dicts(all_lm_scores)
    if run_judge_eval:
        final_results["reference_results"] = _mean_numeric_dicts(all_judge_scores)

    suite.dirs.reference_model_summary_eval.mkdir(parents=True, exist_ok=True)
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
