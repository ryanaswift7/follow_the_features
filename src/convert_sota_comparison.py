"""Convert an averaged SOTA comparison JSON into a main-style CSV table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


HMEAN_METRIC = "alpacaeval_leaderboard_hmean"


def _harmonic_mean(left, right):
    if pd.isna(left) or pd.isna(right):
        return float("nan")
    if left + right == 0:
        return 0.0
    return 2 * left * right / (left + right)


def _flatten_results(entry: dict) -> dict:
    results = entry.get("results", {})
    lm_eval = results.get("lm_eval_results", {}) or {}
    reference = results.get("reference_results", {}) or {}

    row = {
        "model_type": "sota",
        "method": entry.get("method"),
        "topk_features": "N/A",
        "dataset_size": entry.get("budget"),
        "seed_count": len(entry.get("seeds", [])),
        "seeds": ",".join(str(seed) for seed in entry.get("seeds", [])),
    }
    row.update(reference)
    row.update({
        key: value * 100 if isinstance(value, (int, float)) and not isinstance(value, bool) else value
        for key, value in lm_eval.items()
    })
    row[HMEAN_METRIC] = _harmonic_mean(
        row.get("alpacaeval"), row.get("leaderboard")
    )
    return row


def convert_sota_comparison(input_path: Path, output_path: Path | None = None) -> Path:
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    with input_path.open() as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        payload = payload.get("results", [])
    if not isinstance(payload, list):
        raise ValueError("SOTA comparison JSON must contain a list of result rows")

    table = pd.DataFrame([_flatten_results(entry) for entry in payload])
    if table.empty:
        raise ValueError("SOTA comparison JSON contains no result rows")

    metadata = ["model_type", "method", "topk_features", "dataset_size", "seed_count", "seeds"]
    metrics = [column for column in table.columns if column not in metadata]
    table = table[metadata + sorted(metrics)]
    table = table.sort_values(["method", "dataset_size"]).reset_index(drop=True)

    output_path = Path(output_path) if output_path is not None else input_path.with_name("comparison.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert an averaged SOTA comparison JSON to a main-style CSV."
    )
    parser.add_argument("input_path", type=Path, help="Path to comparison.json")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV path; defaults to comparison.csv beside the input JSON",
    )
    args = parser.parse_args()
    output_path = convert_sota_comparison(args.input_path, args.output)
    print(f"Wrote SOTA comparison CSV to {output_path}")


if __name__ == "__main__":
    main()
