from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from src.convert_global_summary import convert_global_summary


class ConvertGlobalSummaryTests(unittest.TestCase):
    def _legacy_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "model_type": "base",
                    "topk_features": -1,
                    "dataset_size": -1,
                    "alpacaeval": 30.0,
                    "leaderboard": 0.2,
                    "ifeval": 0.1,
                },
                {
                    "model_type": "adapter",
                    "topk_features": 10,
                    "dataset_size": 2,
                    "alpacaeval": 50.0,
                    "leaderboard": 0.6,
                    "ifeval": 0.5,
                },
            ]
        )

    def test_csv_conversion_uses_hmean_suffix_and_preserves_input(self):
        with TemporaryDirectory() as directory:
            input_path = Path(directory) / "global_summary.csv"
            self._legacy_table().to_csv(input_path, index=False)

            output_path = convert_global_summary(input_path)

            self.assertEqual(output_path, Path(directory) / "global_summary_hmean.csv")
            converted = pd.read_csv(output_path)
            self.assertEqual(list(converted.columns)[3], "alpacaeval_leaderboard_hmean")
            self.assertEqual(converted.loc[0, "leaderboard"], 20.0)
            self.assertEqual(converted.loc[1, "ifeval"], 50.0)
            self.assertAlmostEqual(converted.loc[1, "alpacaeval_leaderboard_hmean"], 54.54545, places=4)
            self.assertNotIn("alpacaeval_leaderboard_hmean", pd.read_csv(input_path).columns)

    def test_parquet_conversion(self):
        with TemporaryDirectory() as directory:
            input_path = Path(directory) / "global_summary.parquet"
            self._legacy_table().to_parquet(input_path, index=False)

            output_path = convert_global_summary(input_path)

            self.assertEqual(output_path.suffix, ".parquet")
            converted = pd.read_parquet(output_path)
            self.assertEqual(converted.loc[0, "leaderboard"], 20.0)
            self.assertEqual(list(converted.columns)[3], "alpacaeval_leaderboard_hmean")
