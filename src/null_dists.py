import argparse
import gc
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.trainer.sft_trainer import SFTTrainer

from src.eval import eval_and_save
from src.prune import get_formatting_function
from src.schemas.suite import SuiteConfig
from src.sft import PEFT_CONFIG, generate_sft_trainer_args
from src.utils.enums import ModelType
from src.utils.reproducibility import enforce_reproducibility


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate null distribution for experiment suite based on configuration file."
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/base.yaml",
        help="Path to the experiment configuration YAML file (default: 'configs/base.yaml').",
    )
    parser.add_argument(
        "--num-trials",
        "-n",
        type=int,
        default=10,
        help="Number of trials to generate null distributions (default: 10).",
    )
    return parser.parse_args()


def analyze_null_dists(
    null_dist_df: pd.DataFrame,
    output_dir: Union[str, Path],
    experiments_global_summary_df: Optional[pd.DataFrame] = None,
):
    """
    Analyzes Null Distribution vs. Experiments using the ModelType Enum.

    Outputs:
    1. plot_global_scatter.png: Scatter plot of all data points.
    2. null_dist_stats.{md, parquet}: Summary stats of the null distribution itself.
    3. significance_stats.{md, parquet}: Z-scores and P-values of experiments vs null dist.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n[ANALYSIS] Starting Null Distribution Analysis")
    print(f"  - Output Directory: {output_dir}")
    print(f"  - Null Samples: {len(null_dist_df)}")

    # --- 0. Data Cleaning ---
    print("[ANALYSIS] Standardizing column names...")
    if (
        "average_eval_score" in null_dist_df.columns
        and "score" not in null_dist_df.columns
    ):
        null_dist_df = null_dist_df.rename(columns={"average_eval_score": "score"})

    if experiments_global_summary_df is not None:
        if (
            "average_eval_score" in experiments_global_summary_df.columns
            and "score" not in experiments_global_summary_df.columns
        ):
            experiments_global_summary_df = experiments_global_summary_df.rename(
                columns={"average_eval_score": "score"}
            )

    # --- 1. Generate Null Distribution Statistics ---
    print("[ANALYSIS] Calculating Null Distribution Summary Stats...")
    null_stats = (
        null_dist_df.groupby("dataset_size")["score"]
        .agg(
            Count="count",
            Mean="mean",
            Std="std",
            Min="min",
            Q1=lambda x: x.quantile(0.25),
            Median="median",
            Q3=lambda x: x.quantile(0.75),
            Max="max",
        )
        .reset_index()
    )

    # Save Null Stats
    null_stats.to_parquet(output_dir / "null_dist_stats.parquet")
    with open(output_dir / "null_dist_stats.md", "w") as f:
        f.write("# Null Distribution Summary Statistics\n\n")
        f.write(null_stats.to_markdown(index=False, floatfmt=".4f"))
    print(f"  > Saved null stats to {output_dir / 'null_dist_stats.md'}")

    # --- 2. Generate Significance Statistics (Experiments vs Null) ---
    print("[ANALYSIS] Calculating Significance (Z-Scores & P-Values)...")

    sig_rows = []
    baselines = []

    # Pre-calculate null stats dict for fast lookup
    null_lookup = {}
    unique_sizes = null_dist_df["dataset_size"].unique()
    for size in unique_sizes:
        subset = null_dist_df[null_dist_df["dataset_size"] == size]["score"]
        null_lookup[size] = {
            "mean": subset.mean(),
            "std": subset.std(),
            "data": subset.values,
        }

    if experiments_global_summary_df is not None:
        # Extract Baselines
        mask_baseline = (
            experiments_global_summary_df["model_type"] != ModelType.ADAPTER.value
        )
        if mask_baseline.any():
            baselines = experiments_global_summary_df[mask_baseline].to_dict("records")

        # Extract Experiments
        mask_adapter = (
            experiments_global_summary_df["model_type"] == ModelType.ADAPTER.value
        )
        experiments = experiments_global_summary_df[mask_adapter].to_dict("records")

        # 2a. Calculate Adapter Significance
        # Wrapping in tqdm for progress visibility
        if experiments:
            for exp in tqdm(experiments, desc="Processing Adapters", unit="exp"):
                size = exp.get("dataset_size")
                score = exp.get("score")
                topk = exp.get("topk_features")

                if size in null_lookup:
                    stats = null_lookup[size]
                    # Z-Score
                    z_score = (
                        (score - stats["mean"]) / stats["std"]
                        if stats["std"] > 0
                        else np.nan
                    )
                    # Empirical P-Value
                    p_val = (stats["data"] >= score).mean()

                    sig_rows.append(
                        {
                            "Model Type": ModelType.ADAPTER.value,
                            "Label": f"Top-{topk}",
                            "Dataset Size": size,
                            "Score": score,
                            "Null Mean": stats["mean"],
                            "Null Std": stats["std"],
                            "Z-Score": z_score,
                            "Empirical P-Value": p_val,
                        }
                    )

        # 2b. Calculate Baseline Significance
        # Baselines are compared against EVERY dataset size
        if baselines:
            for base in tqdm(baselines, desc="Processing Baselines", unit="base"):
                b_score = base.get("score")
                b_name = base.get("model_type")

                for size, stats in null_lookup.items():
                    z_score = (
                        (b_score - stats["mean"]) / stats["std"]
                        if stats["std"] > 0
                        else np.nan
                    )
                    p_val = (stats["data"] >= b_score).mean()

                    sig_rows.append(
                        {
                            "Model Type": b_name,
                            "Label": b_name,
                            "Dataset Size": size,
                            "Score": b_score,
                            "Null Mean": stats["mean"],
                            "Null Std": stats["std"],
                            "Z-Score": z_score,
                            "Empirical P-Value": p_val,
                        }
                    )

    # Save Significance DF
    sig_df = pd.DataFrame(sig_rows)
    if not sig_df.empty:
        sig_df = sig_df.sort_values(
            ["Dataset Size", "Z-Score"], ascending=[True, False]
        )

        sig_df.to_parquet(output_dir / "significance_stats.parquet")
        with open(output_dir / "significance_stats.md", "w") as f:
            f.write("# Experiment Significance vs Null Distribution\n\n")
            f.write(sig_df.to_markdown(index=False, floatfmt=".4f"))
        print(f"  > Saved significance stats to {output_dir / 'significance_stats.md'}")
    else:
        print("  > No significance data generated (missing experiments?).")

    # --- 3. Plotting (Global Scatter) ---
    print("[ANALYSIS] Generating Global Scatter Plot...")
    plt.figure(figsize=(12, 8))

    # A. Plot Null Data
    null_plot_data = null_dist_df.copy()
    sns.scatterplot(
        data=null_plot_data,
        x="dataset_size",
        y="score",
        color="lightgrey",
        s=60,
        alpha=0.6,
        label="Null Distribution",
        zorder=1,
    )

    # B. Plot Experiment Data (Adapters)
    if experiments_global_summary_df is not None:
        mask_adapter = (
            experiments_global_summary_df["model_type"] == ModelType.ADAPTER.value
        )
        if mask_adapter.any():
            exp_plot_data = experiments_global_summary_df[mask_adapter].copy()
            exp_plot_data["label_col"] = exp_plot_data["topk_features"].apply(
                lambda x: f"Top-{int(x)}"
            )

            unique_labels = sorted(
                exp_plot_data["label_col"].unique(), key=lambda x: int(x.split("-")[1])
            )

            sns.scatterplot(
                data=exp_plot_data,
                x="dataset_size",
                y="score",
                hue="label_col",
                hue_order=unique_labels,
                palette="viridis",
                style="label_col",
                s=120,
                alpha=0.9,
                edgecolor="black",
                zorder=10,
            )

    # C. Plot Baselines (Horizontal Lines)
    baseline_colors = ["red", "blue", "green", "orange"]
    for i, base in enumerate(baselines):
        val = base.get("score", 0)
        label = base.get("model_type", "Baseline")
        color = baseline_colors[i % len(baseline_colors)]

        plt.axhline(
            y=val,
            color=color,
            linestyle="--",
            linewidth=1.5,
            label=f"{label} (Baseline)",
            zorder=5,
        )

    # Formatting
    plt.title("Score vs Dataset Size: Null Distribution vs Experiments")
    plt.xlabel("Dataset Size")
    plt.ylabel("Average Eval Score")
    plt.grid(True, linestyle=":", alpha=0.5)

    unique_sizes = sorted(null_dist_df["dataset_size"].unique())
    plt.xticks(unique_sizes)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    plt.tight_layout()

    plot_path = output_dir / "plot_global_scatter.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"  > Saved plot to {plot_path}")
    print("[ANALYSIS] Complete.\n")


def generate_null_distributions(suite: SuiteConfig, num_trials: int):
    # --- 1. Setup & Logging ---
    dataset_id = suite.prune.cfg.dataset_id
    batch_size = suite.prune.params.batch_size
    num_samples = suite.prune.cfg.num_samples
    seed = suite.seed
    base_model_id = suite.sft.cfg.base_model_id
    dataset_sizes = suite.prune.params.dataset_sizes

    print(f"\n{'=' * 50}")
    print("[NULL-DIST] Initializing Null Distribution Generator")
    print(f"  - Dataset:      {dataset_id}")
    print(f"  - Base Model:   {base_model_id}")
    print(f"  - Dataset Sizes: {dataset_sizes}")
    print(f"  - Trials/Size:  {num_trials}")
    print(f"  - Total Runs:   {len(dataset_sizes) * num_trials}")
    print(f"{'=' * 50}\n")

    # --- 2. Data Loading ---
    print(f"[NULL-DIST] Loading tokenizer: {suite.sft.cfg.tokenizer_id}...")
    tokenizer = AutoTokenizer.from_pretrained(suite.sft.cfg.tokenizer_id)

    print(f"[NULL-DIST] Loading dataset: {dataset_id}...")
    full_ds = load_dataset(dataset_id, split="train")

    if num_samples is not None and num_samples < len(full_ds):
        print(f"[NULL-DIST] Subsampling dataset to {num_samples} rows...")
        full_ds = full_ds.select(range(num_samples))

    print(f"[NULL-DIST] Applying chat formatting (Batch Size: {batch_size})...")
    formatter_func = get_formatting_function(dataset_id)

    # 'desc' adds a progress bar to the map operation
    full_ds = full_ds.map(
        lambda examples: formatter_func(tokenizer, examples, is_inference=False),
        batched=True,
        batch_size=batch_size,
        desc="Tokenizing Data",
    )
    full_ds = full_ds.map(
        format_to_conversation,
        remove_columns=full_ds.column_names,
        # batched=True,
        # batch_size=batch_size,
        # desc="Formatting to Conversation",
    )

    # --- 3. Main Experiment Loop ---
    data = {"dataset_size": [], "trial": [], "score": []}

    print("\n[NULL-DIST] Starting Training Loops...")

    # Outer Progress Bar: Dataset Sizes
    pbar_sizes = tqdm(dataset_sizes, desc="Dataset Sizes", position=0)

    for n in pbar_sizes:
        pbar_sizes.set_description(f"Processing Size: {n}")

        # Inner Progress Bar: Trials
        pbar_trials = tqdm(
            range(num_trials), desc=f"Size {n} Trials", position=1, leave=False
        )

        for i in pbar_trials:
            trial_seed = seed + i
            pbar_trials.set_description(f"Size {n} | Trial {i + 1}/{num_trials}")

            # A. Prepare Dataset (Shuffle & Select)
            dataset = full_ds.shuffle(seed=trial_seed)
            dataset = dataset.select(range(n))

            # B. Load Model
            # Using 'quiet' loading if possible, or just standard
            model = AutoModelForCausalLM.from_pretrained(
                base_model_id, device_map="auto", torch_dtype=torch.bfloat16
            )

            # C. Train
            # Disable trainer output to keep progress bars clean (optional)
            trainer_args = generate_sft_trainer_args(suite=suite, output_dir=None)
            trainer_args.disable_tqdm = True  # Prevent nested tqdm conflict

            trainer = SFTTrainer(
                model=model,
                args=trainer_args,
                train_dataset=dataset,
                peft_config=PEFT_CONFIG,
                processing_class=tokenizer,
            )

            # pbar_trials.write(f"  > Training (Size {n}, Trial {i})...")
            trainer.train()  # Assuming .train() is called implicitly or via SFTTrainer usage

            # D. Evaluate
            # pbar_trials.write(f"  > Evaluating...")

            all_eval_scores, _ = eval_and_save(
                model=trainer.model,
                tokenizer=tokenizer,
                suite=suite,
                model_type=ModelType.ADAPTER,
                dataset_size=n,
                topk=None,
                save_results_and_config=False,
            )
            avg_eval_score = all_eval_scores["average_eval_score"]

            # E. Store Results
            data["dataset_size"].append(n)
            data["trial"].append(i)
            data["score"].append(avg_eval_score)

            # Update progress bar with latest score
            pbar_trials.set_postfix({"Last Score": f"{avg_eval_score:.4f}"})

            # F. Cleanup Memory (Important for loops!)
            del model, trainer
            gc.collect()
            torch.cuda.empty_cache()

    print("\n[NULL-DIST] Generation Complete.")
    return pd.DataFrame(data)


def generate_and_analyze_null_dists(
    suite: SuiteConfig,
    num_trials: int,
    experiments_global_summary_df: Optional[pd.DataFrame] = None,
):

    suite.dirs.null_dist_results.mkdir(parents=True, exist_ok=True)
    enforce_reproducibility(seed=suite.seed)
    null_dist_df = generate_null_distributions(suite=suite, num_trials=num_trials)
    print(null_dist_df)

    null_dist_df.to_markdown(
        suite.dirs.null_dist_results / "raw_null_distribution_data.md"
    )
    null_dist_df.to_parquet(
        suite.dirs.null_dist_results / "raw_null_distribution_data.parquet"
    )

    analyze_null_dists(
        null_dist_df=null_dist_df,
        output_dir=suite.dirs.null_dist_results / "analysis",
        experiments_global_summary_df=experiments_global_summary_df,
    )


def main():
    args = parse_args()
    config_path = args.config

    suite = SuiteConfig.from_yaml(config_path)
    generate_and_analyze_null_dists(suite=suite, num_trials=args.num_trials)


if __name__ == "__main__":
    main()
