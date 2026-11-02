import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import yaml

FEATURE_COLUMN = "feature_idx"
REPO_ROOT = Path(__file__).resolve().parents[2]


def compare_top_k_spearman_large_scale(
    csv_paths, top_k_values, labels=None, feature_col="feature_idx"
):
    """Compare ranked feature reports using a penalized Spearman correlation."""
    if labels is None:
        labels = csv_paths
    elif len(labels) != len(csv_paths):
        raise ValueError("The number of labels must exactly match the number of csv_paths.")

    dfs = {}
    for path, label in zip(csv_paths, labels):
        try:
            df = pd.read_csv(path, usecols=[feature_col], comment="#")
            df["rank"] = np.arange(1, len(df) + 1, dtype=np.int32)
            dfs[label] = df
            print(f"Successfully loaded {len(df):,} ranked features from '{label}'")
        except ValueError:
            print(f"Error: Column '{feature_col}' not found in {path}. Skipping.")
        except Exception as exc:
            print(f"Error loading {path}: {exc}")

    results = []
    for label1, label2 in combinations(dfs.keys(), 2):
        df1 = dfs[label1]
        df2 = dfs[label2]

        for k in top_k_values:
            if k == "all":
                top_df1 = df1
                top_df2 = df2
                penalty_rank = max(len(df1), len(df2)) + 1
            else:
                top_df1 = df1.head(k).copy()
                top_df2 = df2.head(k).copy()
                penalty_rank = k + 1

            merged = pd.merge(
                top_df1,
                top_df2,
                on=feature_col,
                how="outer",
                suffixes=("_1", "_2"),
            )
            merged["rank_1"] = merged["rank_1"].fillna(penalty_rank).astype(np.int32)
            merged["rank_2"] = merged["rank_2"].fillna(penalty_rank).astype(np.int32)

            union_size = len(merged)
            if union_size > 1:
                rho, p_value = spearmanr(merged["rank_1"], merged["rank_2"])
            else:
                rho, p_value = float("nan"), float("nan")

            results.append(
                {
                    "dataset_1": label1,
                    "dataset_2": label2,
                    "k": k,
                    "spearman_rho": rho,
                    "p_value": p_value,
                    "union_size": union_size,
                }
            )

    return pd.DataFrame(results)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare existing feature rankings.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML config containing labeled existing feature reports.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> tuple[list[Path], list[str], list[int | str], Path]:
    with config_path.open() as file:
        config = yaml.safe_load(file) or {}

    reports = config.get("reports")
    if not isinstance(reports, list) or len(reports) < 2:
        raise ValueError("Config must contain at least two reports.")

    labels = [report.get("label") for report in reports]
    if any(not isinstance(label, str) or not label for label in labels):
        raise ValueError("Every report must have a non-empty string label.")
    if len(set(labels)) != len(labels):
        raise ValueError("Report labels must be unique.")

    report_paths = []
    for report in reports:
        path_value = report.get("path")
        if not isinstance(path_value, str) or not path_value:
            raise ValueError("Every report must have a path.")
        path = Path(path_value)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.is_file():
            raise FileNotFoundError(f"Feature report does not exist: {path}")
        columns = pd.read_csv(path, comment="#", nrows=0).columns
        if FEATURE_COLUMN not in columns:
            raise ValueError(f"Feature report is missing '{FEATURE_COLUMN}': {path}")
        report_paths.append(path)

    top_k_values = config.get("top_k_values", [5, 10, 50, 100, 200, 500, 1000, "all"])
    if not isinstance(top_k_values, list) or not top_k_values:
        raise ValueError("top_k_values must be a non-empty list.")
    for value in top_k_values:
        if value != "all" and (not isinstance(value, int) or isinstance(value, bool) or value <= 0):
            raise ValueError("top_k_values must contain positive integers or 'all'.")

    output_dir = Path(config.get("output_dir", "feature_ranking_correlation"))
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    return report_paths, labels, top_k_values, output_dir


def main():
    args = parse_args()
    ranking_paths, labels, top_k_values, output_dir = load_config(args.config)

    results_df = compare_top_k_spearman_large_scale(
        csv_paths=ranking_paths,
        top_k_values=top_k_values,
        labels=labels,
        feature_col=FEATURE_COLUMN,
    )
    if results_df.empty:
        raise RuntimeError("No feature rankings were available for correlation analysis.")

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "feature_ranking_correlation_results.csv"
    markdown_path = output_dir / "feature_ranking_correlation_results.md"
    results_df.to_csv(csv_path, index=False)
    markdown_path.write_text(results_df.to_markdown(index=False) + "\n")

    print("\nResults:")
    print(results_df.to_string(index=False))
    print(f"\nSaved correlation CSV to {csv_path}")
    print(f"Saved correlation Markdown to {markdown_path}")


if __name__ == "__main__":
    main()
