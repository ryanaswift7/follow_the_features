import argparse
import json
from pathlib import Path

from src.analysis import aggregate_results, plot_experiment_results
from src.eval import run_baseline_evals, run_eval
from src.prune import run_prune
from src.schemas.suite import SuiteConfig
from src.sft import run_sft
from src.utils.constants import (
    DATASET_CONFIG_FILENAME,
    EVAL_CONFIG_FILENAME,
    EVAL_ID_KEY,
    EVAL_RESULTS_FILENAME,
    EVAL_RESULTS_KEY,
    MODEL_CONFIG_FILENAME,
    MODEL_EXISTENCE_FILENAME,
    PRUNE_ID_KEY,
    SFT_ID_KEY,
)
from src.utils.enums import ModelType
from src.utils.reproducibility import enforce_reproducibility


def run_experiments(suite: SuiteConfig):
    print(f"[RUN] Starting experiment sweep for suite: {suite.title}")

    run_lm_eval = suite.eval.cfg.lm_eval.run
    run_judge_eval = suite.eval.cfg.alpacaeval.run

    # Step 1: Baselines
    if run_lm_eval or run_judge_eval:
        run_baseline_evals(suite)
    else:
        print("[BASELINE] Both LM eval and judge eval are disabled. Skipping baseline evaluations.")

    # Step 2: Prune, SFT, Eval loop
    expected_result_keys = []
    if run_lm_eval:
        expected_result_keys.append("lm_eval_results")
    if run_judge_eval:
        expected_result_keys.append("reference_results")

    if run_lm_eval or run_judge_eval:
        for k in suite.prune.params.topk_features:
            for n in suite.prune.params.dataset_sizes:
                print(f"[EXPERIMENT] Processing dataset size: {n}, top-k features: {k}")

                eval_results_path = (
                    suite.dirs.get_trained_model_eval(dataset_size=n, topk=k)
                    / EVAL_RESULTS_FILENAME
                )
                eval_config_path = (
                    suite.dirs.get_trained_model_eval(dataset_size=n, topk=k)
                    / EVAL_CONFIG_FILENAME
                )
                need_eval = True
                if eval_results_path.exists() and eval_config_path.exists():
                    with open(eval_results_path, "r") as f:
                        results = json.load(f)
                    with open(eval_config_path, "r") as f:
                        cfg = json.load(f)
                    prune_id_match = cfg.get(PRUNE_ID_KEY) == suite.ids.prune
                    sft_id_match = cfg.get(SFT_ID_KEY) == suite.ids.sft
                    eval_id_match = cfg.get(EVAL_ID_KEY) == suite.ids.eval
                    all_id_match = prune_id_match and sft_id_match and eval_id_match
                    if all_id_match and all(
                        results.get(result_key) is not None for result_key in expected_result_keys
                    ):
                        need_eval = False

                if need_eval:
                    print(
                        f"[MISSING] Results missing or config hashes do not match for model {suite.dirs.get_trained_model_root(dataset_size=n, topk=k).name}. Will run necessary steps."
                    )
                    need_prune = True

                    ds_dir = suite.dirs.get_dataset_path(dataset_size=n, topk=k)
                    ds_cfg_file = ds_dir / DATASET_CONFIG_FILENAME

                    if ds_dir.exists() and ds_cfg_file.exists():
                        with open(ds_cfg_file, "r") as f:
                            ds_cfg = json.load(f)

                        if ds_cfg.get(PRUNE_ID_KEY) == suite.ids.prune:
                            need_prune = False

                    need_sft = True

                    model_root = suite.dirs.get_trained_model_root(dataset_size=n, topk=k)
                    model_cfg_file = model_root / MODEL_CONFIG_FILENAME
                    model_exists_marker = (model_root / MODEL_EXISTENCE_FILENAME).exists()

                    if model_exists_marker and model_cfg_file.exists():
                        with open(model_cfg_file, "r") as f:
                            model_cfg = json.load(f)

                        matches_prune = model_cfg.get(PRUNE_ID_KEY) == suite.ids.prune
                        matches_sft = model_cfg.get(SFT_ID_KEY) == suite.ids.sft

                        if matches_prune and matches_sft:
                            need_sft = False
                    if need_prune:
                        print(
                            f"[MISSING] Dataset missing for model {suite.dirs.get_trained_model_root(dataset_size=n, topk=k).name}. Running dataset pruning..."
                        )
                        run_prune(suite=suite, dataset_size=n, topk=k)
                    else:
                        print(
                            f"[FOUND] Dataset exists for model {suite.dirs.get_trained_model_root(dataset_size=n, topk=k).name}."
                        )

                    if need_sft:
                        print(
                            f"[MISSING] Model missing for model {suite.dirs.get_trained_model_root(dataset_size=n, topk=k).name}. Running SFT..."
                        )
                        run_sft(suite=suite, dataset_size=n, topk=k)
                    else:
                        print(
                            f"[FOUND] Model exists for model {suite.dirs.get_trained_model_root(dataset_size=n, topk=k).name}."
                        )

                    print(
                        f"[EVAL] Running evaluation for model {suite.dirs.get_trained_model_root(dataset_size=n, topk=k).name}..."
                    )
                    run_eval(
                        suite=suite,
                        model_type=ModelType.ADAPTER,
                        dataset_size=n,
                        topk=k,
                    )
                else:
                    print(
                        f"[FOUND] Results exist for model {suite.dirs.get_trained_model_root(dataset_size=n, topk=k).name} and config hashes match."
                    )
    else:
        print("[EXPERIMENT] Both LM eval and judge eval are disabled. Skipping per-model eval loop.")

    experiment_ids = {
        "experiment_id": suite.ids.experiment,
        "prune_id": suite.ids.prune,
        "sft_id": suite.ids.sft,
        "eval_id": suite.ids.eval,
    }

    print(f"[INFO] Relevant Experiment IDs: {experiment_ids}")

    save_path = Path(suite.dirs.main_results) / "suite_ids.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(experiment_ids, f, indent=4)
    print(f"[INFO] Saved experiment IDs to {save_path}")
    return


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run experiment suite based on configuration file."
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        # default="configs/base.yaml",
        required=True,
        help="Path to the experiment configuration YAML file (default: 'configs/base.yaml').",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = args.config

    suite = SuiteConfig.from_yaml(config_path)
    enforce_reproducibility(seed=suite.seed)

    run_experiments(suite)

    # At the end, reload all results to ensure aggregation is complete
    all_results_df = aggregate_results(suite=suite)
    global_summary_df = plot_experiment_results(original_df=all_results_df, suite=suite)

    print("[DONE] All experiments and analysis complete.")


if __name__ == "__main__":
    main()
