import argparse
import json

from src.analysis import aggregate_results, plot_experiment_results, write_manifest
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
    has_current_eval_metrics,
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

    # Step 2: Prune, SFT, Eval loop for every configured seed
    if run_lm_eval or run_judge_eval:
        for seed in suite.seeds:
            seed_suite = suite.for_seed(seed)
            enforce_reproducibility(seed=seed)
            print(f"[SEED] Processing seed {seed}")
            for k in seed_suite.prune.params.topk_features:
                for n in seed_suite.prune.params.dataset_sizes:
                    model_label = seed_suite.dirs.get_trained_model_root(dataset_size=n, topk=k).name
                    print(f"[EXPERIMENT] Processing model: {model_label} (seed: {seed}, dataset size: {n}, top-k features: {k})")

                    eval_results_path = seed_suite.dirs.get_trained_model_eval(dataset_size=n, topk=k) / EVAL_RESULTS_FILENAME
                    eval_config_path = seed_suite.dirs.get_trained_model_eval(dataset_size=n, topk=k) / EVAL_CONFIG_FILENAME
                    need_eval = True
                    if eval_results_path.exists() and eval_config_path.exists():
                        with open(eval_results_path, "r") as f:
                            results = json.load(f)
                        with open(eval_config_path, "r") as f:
                            cfg = json.load(f)
                        ids_match = all(
                            cfg.get(key) == expected
                            for key, expected in (
                                (PRUNE_ID_KEY, seed_suite.ids.prune),
                                (SFT_ID_KEY, seed_suite.ids.sft),
                                (EVAL_ID_KEY, seed_suite.ids.eval),
                            )
                        )
                        if ids_match and cfg.get("seed") == seed and has_current_eval_metrics(
                            results, run_lm_eval, run_judge_eval
                        ):
                            need_eval = False

                    if need_eval:
                        print(f"[MISSING] Results missing for seed {seed}, model {model_label}. Will run necessary steps.")
                        need_prune = True

                        ds_dir = seed_suite.dirs.get_dataset_path(dataset_size=n, topk=k)
                        ds_cfg_file = ds_dir / DATASET_CONFIG_FILENAME

                        if ds_dir.exists() and ds_cfg_file.exists():
                            with open(ds_cfg_file, "r") as f:
                                ds_cfg = json.load(f)

                            if ds_cfg.get(PRUNE_ID_KEY) == seed_suite.ids.prune and ds_cfg.get("seed") == seed:
                                need_prune = False

                        need_sft = True

                        model_root = seed_suite.dirs.get_trained_model_root(dataset_size=n, topk=k)
                        model_cfg_file = model_root / MODEL_CONFIG_FILENAME
                        model_exists_marker = (model_root / MODEL_EXISTENCE_FILENAME).exists()

                        if model_exists_marker and model_cfg_file.exists():
                            with open(model_cfg_file, "r") as f:
                                model_cfg = json.load(f)

                            matches_prune = model_cfg.get(PRUNE_ID_KEY) == seed_suite.ids.prune
                            matches_sft = model_cfg.get(SFT_ID_KEY) == seed_suite.ids.sft
                            matches_seed = model_cfg.get("seed") == seed

                            if matches_prune and matches_sft and matches_seed:
                                need_sft = False
                        if need_prune:
                            print(f"[MISSING] Dataset missing for {ds_dir.name}. Running dataset pruning...")
                            run_prune(suite=seed_suite, dataset_size=n, topk=k)
                        else:
                            print(f"[FOUND] Dataset exists at {ds_dir.name}.")

                        if need_sft:
                            print(f"[MISSING] Model missing for model {model_label}. Running SFT...")
                            run_sft(suite=seed_suite, dataset_size=n, topk=k)
                        else:
                            print(f"[FOUND] Model exists for model {model_label}.")

                        print(f"[EVAL] Running evaluation for model {model_label}...")
                        run_eval(suite=seed_suite, model_type=ModelType.ADAPTER, dataset_size=n, topk=k)
                    else:
                        print(f"[FOUND] Results exist for seed {seed}, model {model_label}.")
    else:
        print("[EXPERIMENT] Both LM eval and judge eval are disabled. Skipping per-model eval loop.")

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
    run_experiments(suite)

    # At the end, reload all results to ensure aggregation is complete
    all_results_df = aggregate_results(suite=suite)
    plot_experiment_results(original_df=all_results_df, suite=suite)
    manifest_path = write_manifest(suite=suite, results_df=all_results_df)
    print(f"[DONE] Manifest saved to {manifest_path}")

    print("[DONE] All experiments and analysis complete.")


if __name__ == "__main__":
    main()
