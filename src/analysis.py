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
            return data  # Return the raw dictionary as-is
        except (json.JSONDecodeError, IOError):
            return None

    # 1. Load Base Model
    base_data = load_json_safe(suite.dirs.base_model_eval / EVAL_RESULTS_FILENAME)
    if base_data:
        # Add metadata for plotting
        base_data.update(
            {
                "model_type": ModelType.BASE.value,
                "topk_features": NAN_RESULT,
                "dataset_size": NAN_RESULT,
            }
        )
        rows.append(base_data)

    # 2. Load Reference Model
    ref_data = load_json_safe(suite.dirs.reference_model_eval / EVAL_RESULTS_FILENAME)
    if ref_data:
        ref_data.update(
            {
                "model_type": ModelType.REFERENCE.value,
                "topk_features": NAN_RESULT,
                "dataset_size": NAN_RESULT,
            }
        )
        rows.append(ref_data)

    # 3. Load Trained Pruned Models
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

    # Convert to DataFrame
    df = pd.DataFrame(rows)
    return df


def plot_experiment_results(
    original_df: pd.DataFrame, suite: SuiteConfig
) -> pd.DataFrame:
    print("[PLOT] Generating individual plots and summary tables per Top-K...")

    # 1. Flatten the nested 'eval_scores' dictionary into columns
    def clean_dict_col(x):
        return x if isinstance(x, dict) else {}

    ref_df = pd.json_normalize(
        original_df["reference_results"].apply(clean_dict_col).tolist()
    )
    ref_df.index = original_df.index
    lm_df = pd.json_normalize(
        original_df["lm_eval_results"].apply(clean_dict_col).tolist()
    )
    lm_df.index = original_df.index
    scores_df = pd.concat(
        [
            original_df.drop(columns=["reference_results", "lm_eval_results"]),
            ref_df,
            lm_df,
        ],
        axis=1,
    )

    # print(scores_df)

    # All metrics available in the data
    all_metrics = list(ref_df.columns) + list(lm_df.columns)

    # Only these metrics will generate .png plots
    PLOT_METRICS = ["average_judge_score", "average_perplexity", "average_lm_eval_score"]

    # 2. Extract Baselines (to be reused as horizontal lines in every plot)
    baseline_df = scores_df[scores_df["model_type"] != ModelType.ADAPTER.value].copy()

    # 3. Get unique Top-K values (ignoring the "Baseline" label if present in this col)
    topk_values = [k for k in scores_df["topk_features"].unique() if k != NAN_RESULT]

    # --- NEW: GLOBAL SUMMARY TABLE (ALL RESULTS) ---
    # Create the root tables directory if it doesn't exist
    global_table_path = suite.dirs.main_results / "tables"
    global_table_path.mkdir(parents=True, exist_ok=True)

    # Prepare Global DataFrame
    global_cols = ["model_type", "topk_features", "dataset_size"] + all_metrics
    global_summary_df = scores_df[global_cols].copy()

    print(global_summary_df)

    # Formatting Clean-up for Baselines in the Global Table
    # is_not_adapter = global_summary_df["model_type"] != ModelType.ADAPTER.value
    # global_summary_df.loc[is_not_adapter, "dataset_size"] = NAN_RESULT
    # global_summary_df.loc[is_not_adapter, "topk_features"] = NAN_RESULT

    # Sort for readability (Model Type -> TopK -> Size)
    global_summary_df = global_summary_df.sort_values(
        by=["model_type", "topk_features", "dataset_size"], na_position="first"
    )

    # Save Global: Parquet
    global_summary_df.to_parquet(global_table_path / "global_summary.parquet")

    # Save Global: CSV (Replaced Markdown)
    global_summary_df.to_csv(global_table_path / "global_summary.csv", index=False)

    print(f"[PLOT] Global summary saved to {global_table_path}")

    # --- EXISTING LOOP: PER TOP-K ---
    for k in topk_values:
        # Create a specific DataFrame for the current Top-K trained models
        pruned_k_df = scores_df[
            (scores_df["model_type"] == ModelType.ADAPTER.value)
            & (scores_df["topk_features"] == k)
        ].copy()
        pruned_k_df = pruned_k_df.sort_values("dataset_size")

        # Create subdirectories
        k_dir = f"top{k}"
        plot_path = suite.dirs.main_results / "plots" / k_dir
        table_path = suite.dirs.main_results / "tables" / k_dir
        plot_path.mkdir(parents=True, exist_ok=True)
        table_path.mkdir(parents=True, exist_ok=True)

        # --- PLOTTING: One plot per specific score ---
        for metric in all_metrics:
            # Check: Is this a metric we want to plot?
            if metric not in PLOT_METRICS:
                continue

            plt.figure(figsize=(10, 6))

            # Draw Horizontal Baseline lines (Base and Reference)
            for _, row in baseline_df.iterrows():
                # Skip if the baseline doesn't have a value for this metric (e.g. Reference missing judge score)
                if pd.isna(row[metric]):
                    continue

                # Check against the Enum value
                is_reference = row["model_type"] == ModelType.REFERENCE.value
                label = f"{row['model_type']} (baseline)"

                # Reference: Dashed (--), Base: Dotted (:)
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

            # Draw the Pruned Model performance line
            plt.plot(
                pruned_k_df["dataset_size"],
                pruned_k_df[metric],
                marker="o",
                linewidth=2,
                label=f"Pruned (Top-{k})",
            )

            # Formatting
            plt.xlabel("Dataset Size")
            plt.ylabel(metric.replace("_", " ").title())
            plt.title(f"{metric.replace('_', ' ').title()} vs. Dataset Size (Top-{k})")
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.grid(True, linestyle=":", alpha=0.6)
            plt.tight_layout()

            # Save individual plot
            plt.savefig(plot_path / f"{metric}.png")
            plt.close()

        # --- TABLE: One comprehensive table for all results in this Top-K ---
        # Combine the baselines and the specific Top-K pruned models
        combined_display_df = pd.concat([baseline_df, pruned_k_df])

        # Select identifier columns and all metric columns
        cols = ["model_type", "dataset_size"] + all_metrics
        final_table = combined_display_df[cols].copy()

        final_table.loc[
            final_table["model_type"] != ModelType.ADAPTER.value, "dataset_size"
        ] = NAN_RESULT

        final_table.to_csv(table_path / "summary_report.csv", index=False)
        final_table.to_parquet(table_path / "summary_report.parquet")

    print(
        f"[PLOT] Completed. Subdirectories created for: {', '.join(map(str, topk_values))}"
    )
    return global_summary_df
