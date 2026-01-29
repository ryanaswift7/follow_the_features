import json
import os

from mvp.eval import eval_run, run_baseline_evals
from mvp.experiment_config_schema import ExperimentConfig
from mvp.prune import run_prune
from mvp.sft import run_sft
from mvp.utils.experiment_helpers import (
    aggregate_results,
    config_hash,
    ensure_dir,
    plot_experiment_results,
)
from mvp.utils.experiment_paths import get_exp_paths


def run_experiments(config):
    print(f"[RUN] Starting experiment sweep for suite: {config.suite_title}")
    # Create a hash for the full experiment config, excluding suite_title and output_dir
    config_for_hash = config.dict().copy() if hasattr(config, "dict") else dict(config)
    config_for_hash.pop("suite_title", None)
    config_for_hash.pop("output_dir", None)
    experiment_id = config_hash(config_for_hash)
    suite_dir = os.path.abspath(os.path.join(config.output_dir, config.suite_title))
    # Use get_exp_paths to get canonical dirs for the first experiment (for dir creation)
    # Create a single consistent timestamp for the whole sweep
    import time

    sweep_timestamp = time.strftime("%Y%m%d_%H%M%S")
    experiment_hash = experiment_id
    results_dir = os.path.join(
        suite_dir, "results", f"{experiment_hash}_{sweep_timestamp}"
    )
    datasets_dir = os.path.join(suite_dir, "datasets")
    models_dir = os.path.join(suite_dir, "models")
    ensure_dir(models_dir)
    ensure_dir(datasets_dir)
    ensure_dir(results_dir)
    results = []
    # Baseline logic (unchanged, but could be refactored to use models_dir)
    baseline_result_path = os.path.join(suite_dir, "baseline_results.json")
    baseline_valid = False
    if os.path.exists(baseline_result_path):
        try:
            with open(baseline_result_path, "r") as f:
                baseline_results = json.load(f)
            eval_hash = config_hash(config.eval.dict())
            # Check base model
            base_model_row = baseline_results.get("base_model")
            base_valid = base_model_row and base_model_row.get("eval_id") == eval_hash
            # If prune model exists, check its eval config too
            prune_model_row = baseline_results.get("prune_model")
            prune_valid = True
            if prune_model_row:
                prune_valid = prune_model_row.get("eval_id") == eval_hash
            baseline_valid = base_valid and prune_valid
        except Exception as e:
            print(
                f"[BASELINE] Error reading baseline results: {e}. Will re-run baselines."
            )
            baseline_valid = False
    if not baseline_valid:
        print("[BASELINE] Running baseline evaluations...")
        run_baseline_evals(config, suite_dir)
    else:
        print(
            "[BASELINE] Baseline results with matching eval config found, skipping baseline evals."
        )
    for top_k in config.prune.top_k_features:
        for threshold in config.prune.prune_thresholds:
            if threshold >= 14.0:
                print(
                    f"[SKIP] Skipping top_k={top_k}, threshold={threshold} due to high threshold."
                )
                continue
            print(f"[EXPERIMENT] Processing top_k={top_k}, threshold={threshold}")
            (
                _results_dir,
                dataset_dir,
                model_dir,
                pruned_path,
                model_subdir,
                eval_subdir,
                _timestamp,
            ) = get_exp_paths(suite_dir, top_k, threshold, config)
            dataset_exists = os.path.exists(pruned_path)
            model_exists = os.path.exists(model_subdir) and any(
                os.scandir(model_subdir)
            )
            eval_path = os.path.join(eval_subdir, "eval_results.json")
            eval_config_path = os.path.join(eval_subdir, "eval_config.json")
            result_row = None
            need_eval = True
            if os.path.exists(eval_path):
                with open(eval_path, "r") as f:
                    row = json.load(f)
                prune_hash = config_hash(config.prune.dict())
                sft_hash = config_hash(config.sft.dict())
                eval_hash = config_hash(config.eval.dict())
                if (
                    row.get("eval_id") == eval_hash
                    and row.get("prune_id") == prune_hash
                    and row.get("sft_id") == sft_hash
                ):
                    print(
                        f"[SKIP] All outputs exist for model {os.path.basename(model_subdir)} and config hashes match, skipping."
                    )
                    result_row = row
                    need_eval = False
            if not dataset_exists:
                print(
                    f"[MISSING] Dataset missing for model {os.path.basename(model_subdir)}. Running prune..."
                )
                run_prune(config, top_k, threshold, pruned_path, dataset_dir)
            else:
                print(
                    f"[FOUND] Dataset exists for model {os.path.basename(model_subdir)}."
                )
            if not model_exists:
                print(
                    f"[MISSING] Model missing for model {os.path.basename(model_subdir)}. Running SFT..."
                )
                ensure_dir(model_subdir)
                run_sft(config, pruned_path, model_subdir, top_k, threshold)
            else:
                print(
                    f"[FOUND] Model exists for model {os.path.basename(model_subdir)}."
                )
            if need_eval:
                print(
                    f"[EVAL] Running evaluation for model {os.path.basename(model_subdir)}..."
                )
                ensure_dir(eval_subdir)
                result_row = eval_run(
                    config=config,
                    adapter_path=model_subdir,
                    result_dir=eval_subdir,
                    top_k=top_k,
                    threshold=threshold,
                    model_path=model_subdir,
                    dataset_path=pruned_path,
                )
                # Save only eval config in eval_config.json
                eval_config = (
                    config.eval.dict()
                    if hasattr(config.eval, "dict")
                    else dict(config.eval)
                )
                with open(eval_config_path, "w") as f:
                    json.dump(eval_config, f, indent=4)
                with open(eval_path, "w") as f:
                    json.dump(result_row, f, indent=4)
                results.append(result_row)
            else:
                # Save only eval config in eval_config.json (if not already saved)
                if not os.path.exists(eval_config_path):
                    eval_config = (
                        config.eval.dict()
                        if hasattr(config.eval, "dict")
                        else dict(config.eval)
                    )
                    with open(eval_config_path, "w") as f:
                        json.dump(eval_config, f, indent=4)
                results.append(result_row)
                print(
                    f"[FOUND] Results exist for model {os.path.basename(model_subdir)} and config hashes match."
                )
    return results, suite_dir, results_dir


def main():
    # Load config from YAML file in the current directory
    config_path = os.path.join(os.path.dirname(__file__), "experiment_config.yaml")
    config = ExperimentConfig.from_yaml(config_path)
    results, suite_dir, results_dir = run_experiments(config)
    # At the end, reload all results to ensure aggregation is complete
    all_results = aggregate_results(suite_dir, config)
    # Save final results with one copy of prune, sft, and eval configs at the top level
    if all_results:
        eval_config_dict = (
            config.eval.dict() if hasattr(config.eval, "dict") else dict(config.eval)
        )
        prune_config_dict = (
            config.prune.dict() if hasattr(config.prune, "dict") else dict(config.prune)
        )
        sft_config_dict = (
            config.sft.dict() if hasattr(config.sft, "dict") else dict(config.sft)
        )
        # Remove per-experiment config copies if present
        for row in all_results:
            row.pop("eval_config", None)
            row.pop("prune_config", None)
            row.pop("sft_config", None)
        final_results = {
            "eval_config": eval_config_dict,
            "prune_config": prune_config_dict,
            "sft_config": sft_config_dict,
            "experiments": all_results,
        }
        final_results_path = os.path.join(results_dir, "final_experiment_results.json")
        os.makedirs(results_dir, exist_ok=True)
        with open(final_results_path, "w") as f:
            json.dump(final_results, f, indent=4)
        print(f"[FINAL] Compiled results saved at {final_results_path}.")
    print("[PLOT] Generating plots and tables...")
    plot_experiment_results(all_results, results_dir)
    print("[DONE] All experiments complete.")


if __name__ == "__main__":
    main()
