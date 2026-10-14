import json
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

import pandas as pd
from datasets import Dataset

from src.analysis import _harmonic_mean, _mean_numeric_dicts
from src.analysis import plot_experiment_results, write_manifest
from src.reference_models import (
    _cache_config_matches,
    _reference_model_manifest_matches,
    _sample_reference_dataset,
)
from src.schemas.suite import SuiteConfig
from src.schemas.sft import SFTConfig
from src.sft import generate_sft_trainer_args


class SeedSweepTests(unittest.TestCase):
    def test_seed_list_requires_unique_values(self):
        config = {
            "title": "test",
            "output_dir": "output",
            "seeds": [3, 7],
            "prune": {
                "cfg": {
                    "prune_model_id": "prune-model",
                    "sae_release": "release",
                    "sae_id": "sae",
                    "dataset_id": "yahma/alpaca-cleaned",
                    "num_samples": 2,
                    "seq_agg_method": "mean",
                    "feat_agg_method": "mean",
                    "divide_by_entropy": False,
                },
                "params": {"batch_size": 1, "dataset_sizes": [2], "topk_features": [1]},
            },
            "sft": {
                "cfg": {
                    "max_seq_length": 16,
                    "learning_rate": 1e-4,
                    "lr_scheduler_type": "cosine",
                    "tokenizer_id": "tokenizer",
                    "base_model_id": "model",
                    "full_ft": False,
                },
                "params": {"batch_size": 1},
            },
            "eval": {"cfg": {}, "params": {"batch_size": 1}},
        }
        suite = SuiteConfig(**config)
        self.assertEqual(suite.seeds, [3, 7])
        self.assertEqual(suite.for_seed(7).seed, 7)

    def test_seed_paths_are_distinct(self):
        config_path = Path("configs/debug.yaml")
        suite = SuiteConfig.from_yaml(config_path)
        with TemporaryDirectory() as output_dir:
            suite = suite.model_copy(update={"output_dir": Path(output_dir)})
            first = suite.for_seed(suite.seeds[0]).dirs.get_trained_model_root(2, 10)
            second = suite.for_seed(suite.seeds[1]).dirs.get_trained_model_root(2, 10)
            self.assertNotEqual(first, second)
            self.assertIn("seed_", first.as_posix())

    def test_reference_paths_are_indexed_by_sft_id(self):
        suite = SuiteConfig.from_yaml("configs/debug.yaml")
        with TemporaryDirectory() as output_dir:
            suite = suite.model_copy(update={"output_dir": Path(output_dir)})
            reference_path = suite.dirs.get_ref_model_root_by_seed(2, suite.seeds[0])

            self.assertEqual(reference_path.parent.parent.name, suite.ids.sft)
            self.assertIn("reference_models", reference_path.parts)
            self.assertIn("ds2", reference_path.parts)
            self.assertIn(f"seed_{suite.seeds[0]}", reference_path.parts)

    def test_nested_scores_are_averaged(self):
        result = _mean_numeric_dicts(
            [
                {"average": 0.2, "nested": {"score": 1.0}},
                {"average": 0.4, "nested": {"score": 3.0}},
            ]
        )
        self.assertEqual(result, {"average": 0.3, "nested": {"score": 2.0}})

    def test_harmonic_mean_uses_0_to_100_scores(self):
        self.assertEqual(_harmonic_mean(50.0, 25.0), 100 / 3)
        self.assertTrue(pd.isna(_harmonic_mean(50.0, float("nan"))))

    def test_analysis_scales_lm_scores_and_writes_hmean_first(self):
        suite = SuiteConfig.from_yaml("configs/debug.yaml")
        with TemporaryDirectory() as output_dir:
            suite = suite.model_copy(update={"output_dir": Path(output_dir)})
            results = pd.DataFrame(
                [
                    {
                        "model_type": "base",
                        "dataset_size": -1,
                        "topk_features": -1,
                        "lm_eval_results": {"leaderboard": 0.2, "ifeval": 0.1},
                        "reference_results": {"alpacaeval": 30.0},
                    },
                    {
                        "model_type": "adapter",
                        "dataset_size": 2,
                        "topk_features": 10,
                        "lm_eval_results": {"leaderboard": 0.6, "ifeval": 0.5},
                        "reference_results": {"alpacaeval": 50.0},
                    },
                ]
            )

            plot_experiment_results(results, suite)

            table_path = suite.dirs.main_results / "tables" / "top10" / "summary_report.csv"
            table = pd.read_csv(table_path)
            self.assertEqual(
                list(table.columns)[2], "alpacaeval_leaderboard_hmean"
            )
            self.assertEqual(table.loc[0, "leaderboard"], 20.0)
            self.assertEqual(table.loc[1, "leaderboard"], 60.0)
            self.assertAlmostEqual(table.loc[1, "alpacaeval_leaderboard_hmean"], 54.54545, places=4)
            self.assertTrue(
                (suite.dirs.main_results / "plots" / "top10" / "alpacaeval_leaderboard_hmean.png").exists()
            )

    def test_manifest_uses_new_results_root_and_relative_model_paths(self):
        suite = SuiteConfig.from_yaml("configs/debug.yaml")
        with TemporaryDirectory() as output_dir:
            suite = suite.model_copy(update={"output_dir": Path(output_dir)})
            results = pd.DataFrame(
                [
                    {
                        "model_type": "base",
                        "dataset_size": -1,
                        "topk_features": -1,
                        "lm_eval_results": {"leaderboard": 0.2, "ifeval": 0.2},
                        "reference_results": {"alpacaeval": 0.3},
                    },
                    {
                        "model_type": "reference",
                        "dataset_size": 2,
                        "topk_features": -1,
                        "lm_eval_results": {"leaderboard": 0.4, "ifeval": 0.4},
                        "reference_results": {"alpacaeval": 0.5},
                    },
                    {
                        "model_type": "adapter",
                        "dataset_size": 2,
                        "topk_features": 10,
                        "lm_eval_results": {"leaderboard": 0.6, "ifeval": 0.6},
                        "reference_results": {"alpacaeval": 0.7},
                    },
                ]
            )
            manifest_path = write_manifest(suite, results)
            manifest = json.loads(manifest_path.read_text())

            self.assertEqual(
                manifest_path,
                Path(output_dir) / "results" / suite.title / suite.ids.experiment / "manifest.json",
            )
            self.assertNotIn("main", manifest_path.parts)
            self.assertEqual(len(manifest["baselines"]), 1)
            self.assertEqual(len(manifest["references"]), 1)
            self.assertEqual(len(manifest["trained_models"]), 1)
            for group in ("baselines", "references", "trained_models"):
                for model in manifest[group]:
                    for model_path in model["model_paths"]:
                        self.assertFalse(Path(model_path).is_absolute())

    def test_scheduler_configuration_reaches_trl(self):
        suite = SuiteConfig.from_yaml("configs/debug.yaml")
        self.assertEqual(suite.sft.cfg.lr_scheduler_type, "cosine")

        linear_suite = suite.model_copy(
            update={
                "sft": suite.sft.model_copy(
                    update={
                        "cfg": suite.sft.cfg.model_copy(
                            update={"lr_scheduler_type": "linear"}
                        )
                    }
                )
            }
        )
        trainer_args = generate_sft_trainer_args(
            suite=linear_suite.for_seed(suite.seeds[0]),
            output_dir=None,
            num_training_examples=2,
        )
        self.assertEqual(trainer_args.lr_scheduler_type, "linear")

    def test_scheduler_configuration_is_validated(self):
        suite = SuiteConfig.from_yaml("configs/debug.yaml")
        with self.assertRaises(ValueError):
            SFTConfig(
                **{
                    **suite.sft.cfg.model_dump(),
                    "lr_scheduler_type": "not-a-scheduler",
                }
            )

    def test_reference_cache_requires_matching_sft_id(self):
        suite = SuiteConfig.from_yaml("configs/debug.yaml")
        cache_config = {"eval_id": suite.ids.eval, "sft_id": suite.ids.sft}

        self.assertTrue(_cache_config_matches(cache_config, suite))
        self.assertTrue(
            _cache_config_matches(
                {**cache_config, "seed": suite.seeds[0]},
                suite,
                seed=suite.seeds[0],
            )
        )
        self.assertFalse(
            _cache_config_matches(
                {**cache_config, "sft_id": "sft_stale"},
                suite,
            )
        )
        self.assertFalse(
            _cache_config_matches(
                {**cache_config, "seed": suite.seeds[0]},
                suite,
                seed=suite.seeds[1],
            )
        )

    def test_reference_model_manifest_requires_matching_sft_id(self):
        suite = SuiteConfig.from_yaml("configs/debug.yaml")
        with TemporaryDirectory() as output_dir:
            model_dir = Path(output_dir)
            manifest_path = model_dir / "reference_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "model_type": "reference",
                        "base_model_id": suite.sft.cfg.base_model_id,
                        "dataset_size": 2,
                        "seed": suite.seeds[0],
                        "sft_id": suite.ids.sft,
                    }
                )
            )

            self.assertTrue(
                _reference_model_manifest_matches(model_dir, suite, 2, suite.seeds[0])
            )
            manifest_path.write_text(
                json.dumps(
                    {
                        "model_type": "reference",
                        "base_model_id": suite.sft.cfg.base_model_id,
                        "dataset_size": 2,
                        "seed": suite.seeds[0],
                        "sft_id": "sft_stale",
                    }
                )
            )
            self.assertFalse(
                _reference_model_manifest_matches(model_dir, suite, 2, suite.seeds[0])
            )

    def test_reference_dataset_is_shuffled_before_sampling(self):
        raw_dataset = Dataset.from_dict({"id": list(range(20))})
        sampled = _sample_reference_dataset(raw_dataset, num_samples=5, seed=7)
        expected = raw_dataset.shuffle(seed=7).select(range(5))

        self.assertEqual(sampled["id"], expected["id"])
        self.assertNotEqual(sampled["id"], list(range(5)))

    def test_fine_tuning_mode_is_required(self):
        suite = SuiteConfig.from_yaml("configs/debug.yaml")
        config = suite.sft.cfg.model_dump()
        config.pop("full_ft")
        with self.assertRaises(ValueError):
            SFTConfig(**config)


if __name__ == "__main__":
    unittest.main()
