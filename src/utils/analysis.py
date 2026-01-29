import json

import matplotlib.pyplot as plt
import pandas as pd

from src.suite_cfg_schema import SuiteConfig
from src.utils.constants import EVAL_RESULTS_FILENAME, NA_RESULT_STR
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
                "topk_features": NA_RESULT_STR,
                "dataset_size": NA_RESULT_STR,
            }
        )
        rows.append(base_data)

    # 2. Load Supervisor Model
    sup_data = load_json_safe(suite.dirs.supervisor_model_eval / EVAL_RESULTS_FILENAME)
    if sup_data:
        sup_data.update(
            {
                "model_type": ModelType.SUPERVISOR.value,
                "topk_features": NA_RESULT_STR,
                "dataset_size": NA_RESULT_STR,
            }
        )
        rows.append(sup_data)

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


def plot_experiment_results(original_df: pd.DataFrame, suite: SuiteConfig):
    print("[PLOT] Generating individual plots and summary tables per Top-K...")
    print(original_df)

    # 1. Flatten the nested 'eval_scores' dictionary into columns
    scores_df = pd.json_normalize(original_df["eval_scores"])
    df = pd.concat([original_df.drop(columns=["eval_scores"]), scores_df], axis=1)

    # All metrics available to plot (e.g., 'average_eval_score', 'task_1', 'task_2')
    metrics = list(scores_df.columns)

    # 2. Extract Baselines (to be reused as horizontal lines in every plot)
    baseline_df = df[df["model_type"] != ModelType.ADAPTER.value].copy()

    # 3. Get unique Top-K values (ignoring the "Baseline" label)
    topk_values = [k for k in df["topk_features"].unique() if k != NA_RESULT_STR]

    for k in topk_values:
        # Create a specific DataFrame for the current Top-K trained models
        pruned_k_df = df[
            (df["model_type"] == ModelType.ADAPTER.value) & (df["topk_features"] == k)
        ].copy()
        pruned_k_df = pruned_k_df.sort_values("dataset_size")

        # Create subdirectories: results/plots/top{k} and results/tables/top{k}
        k_dir = f"top{k}"
        plot_path = suite.dirs.results / "plots" / k_dir
        table_path = suite.dirs.results / "tables" / k_dir
        plot_path.mkdir(parents=True, exist_ok=True)
        table_path.mkdir(parents=True, exist_ok=True)

        # --- PLOTTING: One plot per score ---
        for metric in metrics:
            plt.figure(figsize=(10, 6))

            # Draw Horizontal Baseline lines (Base and Supervisor)
            for _, row in baseline_df.iterrows():
                # Check against the Enum value
                is_supervisor = row["model_type"] == ModelType.SUPERVISOR.value

                label = f"{row['model_type']} (baseline)"

                # Both use "grey", only the style changes
                # Supervisor: Dashed (--)
                # Base: Dotted (:)
                style = "--" if is_supervisor else ":"

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
        cols = ["model_type", "dataset_size"] + metrics
        final_table = combined_display_df[cols].copy()

        # Formatting: Change '0' dataset size to 'N/A' for baselines
        final_table.loc[
            final_table["model_type"] != ModelType.ADAPTER.value, "dataset_size"
        ] = NA_RESULT_STR

        # Save as Markdown
        with open(table_path / "summary_report.md", "w") as f:
            f.write(f"# Experiment Summary: Top-{k} Features\n\n")
            f.write(final_table.to_markdown(index=False, floatfmt=".4f"))

    print(
        f"[PLOT] Completed. Subdirectories created for: {', '.join(map(str, topk_values))}"
    )
