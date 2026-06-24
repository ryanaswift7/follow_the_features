import json

import matplotlib.pyplot as plt
import pandas as pd

from src.schemas.suite import SuiteConfig
from src.utils.constants import EVAL_RESULTS_FILENAME, NAN_RESULT
from src.utils.enums import ModelType


def aggregate_results(suite: SuiteConfig) -> pd.DataFrame:
    """
    Aggregates results into a long-format DataFrame for easy grouping/plotting.
    Each row contains model identifiers, params (topk, n), and eval metrics.
    """
    print(f"[AGGREGATE] Aggregating results from: {suite.dirs.suite_root}")
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

    base_data = load_json_safe(suite.dirs.base_model_eval / EVAL_RESULTS_FILENAME)
    if base_data:
        base_data.update(
            {
                "model_type": ModelType.BASE.value,
                "topk_features": NAN_RESULT,
                "dataset_size": NAN_RESULT,
            }
        )
        rows.append(base_data)

    ref_data = load_json_safe(suite.dirs.reference_model_summary_eval / EVAL_RESULTS_FILENAME)
    if ref_data:
        ref_data.update(
            {
                "model_type": ModelType.REFERENCE.value,
                "topk_features": NAN_RESULT,
                "dataset_size": NAN_RESULT,
            }
        )
        rows.append(ref_data)

    for k in suite.prune.params.topk_features:
        for n in suite.prune.params.dataset_sizes:
            path = (
                suite.dirs.get_trained_model_eval(dataset_size=n, topk=k)
                / EVAL_RESULTS_FILENAME
            )
            data = load_json_safe(path)
            if data:
                data.update(
                    {
                        "model_type": ModelType.ADAPTER.value,
                        "topk_features": k,
                        "dataset_size": n,
                    }
                )
                rows.append(data)

    return pd.DataFrame(rows)


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
    scores_df = pd.concat(
        [
            original_df.drop(columns=["reference_results", "lm_eval_results"], errors="ignore"),
            ref_df,
            lm_df,
        ],
        axis=1,
    )

    all_metrics = list(ref_df.columns) + list(lm_df.columns)

    if original_df.empty:
        print("[PLOT] No evaluation rows found. Skipping plots and summary generation.")
        return original_df

    PLOT_METRICS = ["average_judge_score", "length_controlled_winrate", "win_rate", "average_perplexity", "average_lm_eval_score"]

    baseline_df = scores_df[scores_df["model_type"] != ModelType.ADAPTER.value].copy()
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
            if metric not in PLOT_METRICS:
                continue

            plt.figure(figsize=(10, 6))
            for _, row in baseline_df.iterrows():
                if pd.isna(row[metric]):
                    continue

                is_reference = row["model_type"] == ModelType.REFERENCE.value
                label = f"{row['model_type']} (baseline)"
                style = "--" if is_reference else ":"

                plt.axhline(
                    y=row[metric],
                    color="grey",
                    linestyle=style,
                    label=label,
                    alpha=0.7,
                    linewidth=1.5,
                    zorder=1,
                )

            plt.plot(
                pruned_k_df["dataset_size"],
                pruned_k_df[metric],
                marker="o",
                linewidth=2,
                label=f"Pruned (Top-{k})",
            )

            plt.xlabel("Dataset Size")
            plt.ylabel(metric.replace("_", " ").title())
            plt.title(f"{metric.replace('_', ' ').title()} vs. Dataset Size (Top-{k})")
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.grid(True, linestyle=":", alpha=0.6)
            plt.tight_layout()
            plt.savefig(plot_path / f"{metric}.png")
            plt.close()

        combined_display_df = pd.concat([baseline_df, pruned_k_df])
        cols = ["model_type", "dataset_size"] + all_metrics
        final_table = combined_display_df[cols].copy()
        final_table.loc[
            final_table["model_type"] != ModelType.ADAPTER.value, "dataset_size"
        ] = NAN_RESULT
        final_table.to_csv(table_path / "summary_report.csv", index=False)
        final_table.to_parquet(table_path / "summary_report.parquet")

    print(f"[PLOT] Completed. Subdirectories created for: {', '.join(map(str, topk_values))}")
    return global_summary_df
