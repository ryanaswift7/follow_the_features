import argparse
from pathlib import Path

import pandas as pd

from src.analysis import HMEAN_METRIC, _harmonic_mean


LM_EVAL_METRICS = (
    "leaderboard",
    "ifeval",
    "bbh",
    "math_hard",
    "gpqa",
    "musr",
    "mmlu_pro",
)
METADATA_COLUMNS = ("model_type", "topk_features", "dataset_size")


def _output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_hmean{input_path.suffix}")


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    raise ValueError("Input must be a .csv or .parquet file")


def _write_table(df: pd.DataFrame, path: Path) -> None:
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    elif path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    else:
        raise ValueError("Output must be a .csv or .parquet file")


def convert_global_summary(input_path: Path, output_path: Path | None = None) -> Path:
    """Convert a legacy global summary table to the 0-100 and hmean format."""
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    output_path = Path(output_path) if output_path is not None else _output_path(input_path)
    if output_path.resolve() == input_path.resolve():
        raise ValueError("Output must be a new file; use the default _hmean suffix")

    table = _read_table(input_path)
    required_columns = {"alpacaeval", "leaderboard"}
    missing = required_columns - set(table.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if HMEAN_METRIC in table.columns:
        raise ValueError(f"Table already contains {HMEAN_METRIC}")

    for metric in LM_EVAL_METRICS:
        if metric not in table.columns:
            continue
        values = pd.to_numeric(table[metric], errors="coerce")
        invalid = values.isna() & table[metric].notna()
        if invalid.any():
            raise ValueError(f"Column {metric!r} contains non-numeric values")
        table[metric] = values * 100

    table[HMEAN_METRIC] = table.apply(
        lambda row: _harmonic_mean(row["alpacaeval"], row["leaderboard"]), axis=1
    )

    metric_start = next(
        (index for index, column in enumerate(table.columns) if column not in METADATA_COLUMNS),
        len(table.columns),
    )
    metric_columns = [column for column in table.columns if column != HMEAN_METRIC]
    table = table[metric_columns[:metric_start] + [HMEAN_METRIC] + metric_columns[metric_start:]]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_table(table, output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a legacy global summary table to 0-100 scores with a harmonic mean."
    )
    parser.add_argument("input_path", type=Path, help="Path to global_summary.csv or global_summary.parquet")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output path; defaults to the input path with a _hmean suffix",
    )
    args = parser.parse_args()
    output_path = convert_global_summary(args.input_path, args.output)
    print(f"Wrote converted global summary to {output_path}")


if __name__ == "__main__":
    main()
