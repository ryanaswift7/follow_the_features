import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.schemas.suite import SuiteConfig
from src.utils.constants import EVAL_RESULTS_FILENAME, NAN_RESULT
from src.utils.enums import ModelType


HMEAN_METRIC = "alpacaeval_leaderboard_hmean"


def _harmonic_mean(alpacaeval: float, leaderboard: float) -> float:
    if pd.isna(alpacaeval) or pd.isna(leaderboard):
        return float("nan")
    if alpacaeval + leaderboard == 0:
        return 0.0
    return 2 * alpacaeval * leaderboard / (alpacaeval + leaderboard)


def _mean_numeric_dicts(dicts: list[dict]) -> dict:
    """Average numeric leaves while preserving nested result structures."""
    if not dicts:
        return {}

    keys = set().union(*(item.keys() for item in dicts))
    result = {}
    for key in sorted(keys):
        values = [item[key] for item in dicts if key in item]
        numeric = [value for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
        nested = [value for value in values if isinstance(value, dict)]
        if numeric:
            result[key] = round(sum(numeric) / len(numeric), 4)
        elif nested:
            result[key] = _mean_numeric_dicts(nested)
    return result


def aggregate_results(suite: SuiteConfig) -> pd.DataFrame:
    """
    Aggregates results into a long-format DataFrame for easy grouping/plotting.
    Each row contains model identifiers, params (topk, n), and eval metrics.
    """
    print(f"[AGGREGATE] Aggregating results from: {suite.dirs.main_results}")
    rows = []

    def load_json_safe(path):
        if not path.exists():
            return None
        try:
            with open(path, "r") as f:
                data = json.load(f)
            return data
        except (json.JSONDecodeError, IOError):
            return None

    base_seed_results = []
    for seed in suite.seeds:
        base_data = load_json_safe(
            suite.dirs.get_base_model_eval_by_seed(seed) / EVAL_RESULTS_FILENAME
        )
        if base_data:
            base_seed_results.append(base_data)
    if base_seed_results:
        base_data = {
            "lm_eval_results": _mean_numeric_dicts(
                [item["lm_eval_results"] for item in base_seed_results if item.get("lm_eval_results")]
            ),
            "reference_results": _mean_numeric_dicts(
                [item["reference_results"] for item in base_seed_results if item.get("reference_results")]
            ),
            "seed_count": len(base_seed_results),
            "seeds": [item.get("seed") for item in base_seed_results],
            "model_type": ModelType.BASE.value,
            "topk_features": NAN_RESULT,
            "dataset_size": NAN_RESULT,
        }
        rows.append(base_data)

    for n in suite.reference_dataset_sizes:
        ref_data = load_json_safe(
            suite.dirs.reference_model_summary_eval(n) / EVAL_RESULTS_FILENAME
        )
        if ref_data:
            ref_data.update(
                {
                    "model_type": ModelType.REFERENCE.value,
                    "topk_features": NAN_RESULT,
                    "dataset_size": n,
                }
            )
            rows.append(ref_data)

    for k in suite.prune.params.topk_features:
        for n in suite.prune.params.dataset_sizes:
            seed_results = []
            for seed in suite.seeds:
                path = suite.for_seed(seed).dirs.get_trained_model_eval(dataset_size=n, topk=k) / EVAL_RESULTS_FILENAME
                data = load_json_safe(path)
                if data:
                    seed_results.append(data)
            if seed_results:
                data = {
                    "lm_eval_results": _mean_numeric_dicts(
                        [item["lm_eval_results"] for item in seed_results if item.get("lm_eval_results")]
                    ),
                    "reference_results": _mean_numeric_dicts(
                        [item["reference_results"] for item in seed_results if item.get("reference_results")]
                    ),
                    "seed_count": len(seed_results),
                    "seeds": [item.get("seed") for item in seed_results],
                    "model_type": ModelType.ADAPTER.value,
                    "topk_features": k,
                    "dataset_size": n,
                }
                rows.append(data)

    return pd.DataFrame(rows)


def write_manifest(suite: SuiteConfig, results_df: pd.DataFrame) -> Path:
    """Write aggregate model scores and relative model locations."""
    project_root = Path(__file__).resolve().parents[1]

    def relative_model_path(path: Path) -> str:
        return os.path.relpath(path.resolve(), project_root).replace(os.sep, "/")

    def scores_for_row(row: pd.Series | None) -> dict:
        if row is None:
            return {}
        return {
            key: row[key]
            for key in ("lm_eval_results", "reference_results")
            if isinstance(row.get(key), dict)
        }

    def find_row(model_type: str, dataset_size=None, topk_features=None):
        if results_df.empty:
            return None
        matches = results_df[results_df["model_type"] == model_type]
        if dataset_size is not None:
            matches = matches[matches["dataset_size"] == dataset_size]
        if topk_features is not None:
            matches = matches[matches["topk_features"] == topk_features]
        return None if matches.empty else matches.iloc[0]

    manifest = {
        "suite": {
            "title": suite.title,
            "seeds": suite.seeds,
            "experiment_id": suite.ids.experiment,
            "prune_id": suite.ids.prune,
            "sft_id": suite.ids.sft,
            "eval_id": suite.ids.eval,
        },
        "baselines": [],
        "references": [],
        "trained_models": [],
    }

    base_row = find_row(ModelType.BASE.value)
    if base_row is not None:
        manifest["baselines"].append(
            {
                "name": Path(suite.sft.cfg.base_model_id).name,
                "model_type": ModelType.BASE.value,
                "seeds": suite.seeds,
                "model_paths": [relative_model_path(suite.dirs.base_model_root)],
                "evaluation_paths": [
                    relative_model_path(suite.dirs.get_base_model_eval_by_seed(seed))
                    for seed in suite.seeds
                ],
                "scores": scores_for_row(base_row),
            }
        )

    for dataset_size in suite.reference_dataset_sizes:
        reference_row = find_row(ModelType.REFERENCE.value, dataset_size=dataset_size)
        if reference_row is None:
            continue
        manifest["references"].append(
            {
                "name": f"reference_ds{dataset_size}_{Path(suite.sft.cfg.base_model_id).name}",
                "model_type": ModelType.REFERENCE.value,
                "dataset_size": dataset_size,
                "seeds": suite.seeds,
                "model_paths": [
                    relative_model_path(
                        suite.dirs.get_ref_model_root_by_seed(dataset_size, seed)
                    )
                    for seed in suite.seeds
                ],
                "scores": scores_for_row(reference_row),
            }
        )

    for topk in suite.prune.params.topk_features:
        for dataset_size in suite.prune.params.dataset_sizes:
            row = find_row(ModelType.ADAPTER.value, dataset_size, topk)
            if row is None:
                continue
            manifest["trained_models"].append(
                {
                    "name": f"ds{dataset_size}_topk{topk}",
                    "model_type": ModelType.ADAPTER.value,
                    "dataset_size": dataset_size,
                    "topk_features": topk,
                    "seeds": suite.seeds,
                    "model_paths": [
                        relative_model_path(
                            suite.for_seed(seed).dirs.get_trained_model_root(
                                dataset_size=dataset_size, topk=topk
                            )
                        )
                        for seed in suite.seeds
                    ],
                    "scores": scores_for_row(row),
                }
            )

    manifest_path = suite.dirs.main_results / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path


def plot_experiment_results(original_df: pd.DataFrame, suite: SuiteConfig) -> pd.DataFrame:
    print("[PLOT] Generating individual plots and summary tables per Top-K...")

    def clean_dict_col(x):
        return x if isinstance(x, dict) else {}

    ref_values = (
        original_df["reference_results"].apply(clean_dict_col).tolist()
        if "reference_results" in original_df.columns
        else [{} for _ in range(len(original_df))]
    )
    lm_values = (
        original_df["lm_eval_results"].apply(clean_dict_col).tolist()
        if "lm_eval_results" in original_df.columns
        else [{} for _ in range(len(original_df))]
    )

    ref_df = pd.json_normalize(ref_values)
    ref_df.index = original_df.index
    lm_df = pd.json_normalize(lm_values)
    lm_df.index = original_df.index
    for column in lm_df.select_dtypes(include="number").columns:
        lm_df[column] *= 100

    scores_df = pd.concat(
        [
            original_df.drop(columns=["reference_results", "lm_eval_results"], errors="ignore"),
            ref_df,
            lm_df,
        ],
        axis=1,
    )

    scores_df[HMEAN_METRIC] = scores_df.apply(
        lambda row: _harmonic_mean(row.get("alpacaeval"), row.get("leaderboard")),
        axis=1,
    )
    all_metrics = [HMEAN_METRIC] + list(ref_df.columns) + list(lm_df.columns)

    if original_df.empty:
        print("[PLOT] No evaluation rows found. Skipping plots and summary generation.")
        return original_df

    plot_metrics = [
        metric
        for metric in all_metrics
        if metric in {HMEAN_METRIC, "alpacaeval", "leaderboard"} or metric in {
            "ifeval", "bbh", "math_hard", "gpqa", "musr", "mmlu_pro"
        }
    ]

    base_df = scores_df[scores_df["model_type"] == ModelType.BASE.value].copy()
    reference_df = scores_df[
        scores_df["model_type"] == ModelType.REFERENCE.value
    ].copy()
    topk_values = [k for k in scores_df["topk_features"].unique() if k != NAN_RESULT]

    global_table_path = suite.dirs.main_results / "tables"
    global_table_path.mkdir(parents=True, exist_ok=True)

    global_cols = ["model_type", "topk_features", "dataset_size"] + all_metrics
    global_summary_df = scores_df[global_cols].copy()

    print(global_summary_df)

    global_summary_df = global_summary_df.sort_values(
        by=["model_type", "topk_features", "dataset_size"], na_position="first"
    )

    global_summary_df.to_parquet(global_table_path / "global_summary.parquet")
    global_summary_df.to_csv(global_table_path / "global_summary.csv", index=False)

    print(f"[PLOT] Global summary saved to {global_table_path}")

    for k in topk_values:
        pruned_k_df = scores_df[
            (scores_df["model_type"] == ModelType.ADAPTER.value)
            & (scores_df["topk_features"] == k)
        ].copy()
        pruned_k_df = pruned_k_df.sort_values("dataset_size")

        k_dir = f"top{k}"
        plot_path = suite.dirs.main_results / "plots" / k_dir
        table_path = suite.dirs.main_results / "tables" / k_dir
        plot_path.mkdir(parents=True, exist_ok=True)
        table_path.mkdir(parents=True, exist_ok=True)

        for metric in all_metrics:
            if metric not in plot_metrics:
                continue

            plt.figure(figsize=(10, 6))
            for _, row in base_df.iterrows():
                if pd.isna(row[metric]):
                    continue

                label = f"{row['model_type']} (baseline)"

                plt.axhline(
                    y=row[metric],
                    color="grey",
                    linestyle=":",
                    label=label,
                    alpha=0.7,
                    linewidth=1.5,
                    zorder=1,
                )

            reference_metric_df = reference_df.dropna(subset=[metric]).sort_values(
                "dataset_size"
            )
            if not reference_metric_df.empty:
                plt.plot(
                    reference_metric_df["dataset_size"],
                    reference_metric_df[metric],
                    marker="s",
                    linestyle="--",
                    linewidth=1.8,
                    color="grey",
                    label="Reference (Random)",
                    alpha=0.85,
                    zorder=2,
                )

            plt.plot(
                pruned_k_df["dataset_size"],
                pruned_k_df[metric],
                marker="o",
                linewidth=2,
                label=f"Filtered (Top-{k})",
            )

            plt.xlabel("Dataset Size")
            plt.ylabel(metric.replace("_", " ").title())
            plt.title(f"{metric.replace('_', ' ').title()} vs. Dataset Size (Top-{k})")
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.grid(True, linestyle=":", alpha=0.6)
            plt.tight_layout()
            plt.savefig(plot_path / f"{metric}.png")
            plt.close()

        combined_display_df = pd.concat([base_df, reference_df, pruned_k_df])
        cols = ["model_type", "dataset_size"] + all_metrics
        final_table = combined_display_df[cols].copy()
        final_table.loc[
            final_table["model_type"] == ModelType.BASE.value, "dataset_size"
        ] = NAN_RESULT
        final_table.to_csv(table_path / "summary_report.csv", index=False)
        final_table.to_parquet(table_path / "summary_report.parquet")

    print(f"[PLOT] Completed. Subdirectories created for: {', '.join(map(str, topk_values))}")
    return global_summary_df
