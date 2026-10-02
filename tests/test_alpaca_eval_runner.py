import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from src.schemas.eval import AlpacaEvalConfig
from src.utils.alpaca_eval_runner import (
    _filter_alpaca_eval_summary,
    _prepare_eval_cache,
    build_annotators_config,
    build_model_configs,
)


class AlpacaEvalRunnerTests(unittest.TestCase):
    def _suite(self, cfg: AlpacaEvalConfig):
        return SimpleNamespace(
            eval=SimpleNamespace(cfg=SimpleNamespace(alpacaeval=cfg)),
            sft=SimpleNamespace(cfg=SimpleNamespace(base_model_id="base-model")),
        )

    def test_alpacaeval_uses_length_controlled_winrate(self):
        summary = _filter_alpaca_eval_summary(
            {
                "length_controlled_winrate": 0.42,
                "win_rate": 0.99,
                "average_judge_score": 0.99,
            }
        )
        self.assertEqual(summary["alpacaeval"], 0.42)
        self.assertNotIn("win_rate", summary)
        self.assertNotIn("average_judge_score", summary)

    def test_explicit_backends_are_resolved(self):
        cfg = AlpacaEvalConfig(
            candidate_backend="vllm",
            judge_backend="vllm",
            model_kwargs={"tensor_parallel_size": 1},
        )

        self.assertEqual(cfg.effective_candidate_backend, "vllm")
        self.assertEqual(cfg.effective_judge_backend, "vllm")
        self.assertEqual(cfg.effective_judge_model_kwargs, {"tensor_parallel_size": 1})

    def test_default_dtypes_are_candidate_bfloat16_and_judge_native(self):
        cfg = AlpacaEvalConfig()

        self.assertEqual(cfg.candidate_dtype, "bfloat16")
        self.assertEqual(cfg.judge_dtype, "auto")

    def test_candidate_vllm_kwargs_are_written_to_alpaca_config(self):
        cfg = AlpacaEvalConfig(
            candidate_backend="vllm",
            candidate_model_kwargs={"tensor_parallel_size": 2, "max_model_len": 4096},
        )
        suite = self._suite(cfg)

        with patch("src.utils.alpaca_eval_runner.torch.cuda.device_count", return_value=1):
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                prompt_path = tmp_path / "prompt.txt"
                prompt_path.write_text("{instruction}")
                config_dir = build_model_configs(
                    suite=suite,
                    model_name="candidate",
                    prompt_template_path=prompt_path,
                    model_path="local-model",
                    model_configs_dir=tmp_path / "model-config",
                    candidate_backend=cfg.effective_candidate_backend,
                )

                config = yaml.safe_load((config_dir / "configs.yaml").read_text())["candidate"]
                self.assertEqual(config["fn_completions"], "vllm_local_completions")
                self.assertEqual(config["completions_kwargs"]["model_kwargs"]["dtype"], "bfloat16")
                self.assertEqual(config["completions_kwargs"]["model_kwargs"]["tensor_parallel_size"], 1)
                self.assertEqual(config["completions_kwargs"]["model_kwargs"]["max_model_len"], 4096)

    @patch("src.utils.alpaca_eval_runner.torch.cuda.device_count", return_value=3)
    def test_tensor_parallel_size_uses_all_visible_gpus(self, _device_count):
        from src.utils.alpaca_eval_runner import _detected_tensor_parallel_size

        self.assertEqual(_detected_tensor_parallel_size(), 3)

    @patch("src.utils.alpaca_eval_runner.torch.cuda.device_count", return_value=0)
    def test_tensor_parallel_size_requires_a_visible_gpu(self, _device_count):
        from src.utils.alpaca_eval_runner import _detected_tensor_parallel_size

        with self.assertRaisesRegex(RuntimeError, "at least one visible CUDA GPU"):
            _detected_tensor_parallel_size()

    @patch("src.utils.alpaca_eval_runner._alpaca_eval_package_root")
    @patch("src.utils.alpaca_eval_runner.torch.cuda.device_count", return_value=1)
    def test_vllm_judge_parser_ignores_reasoning_tags(self, _device_count, package_root):
        cfg = AlpacaEvalConfig(
            judge_backend="vllm",
            model_name="judge-model",
            judge_model_kwargs={"tensor_parallel_size": 2},
        )
        suite = self._suite(cfg)

        with tempfile.TemporaryDirectory() as tmp:
            package_root.return_value = Path(tmp)
            config_dir = build_annotators_config(suite, Path(tmp) / "eval")
            config = yaml.safe_load((config_dir / "configs.yaml").read_text())
            annotator = config["fdp_alpaca_eval_local_vllm"]

            self.assertEqual(annotator["fn_completion_parser"], "regex_parser")
            self.assertEqual(annotator["completions_kwargs"]["enable_thinking"], False)
            self.assertEqual(annotator["fn_completions"], "vllm_local_completions")
            parser_kwargs = annotator["completion_parser_kwargs"]
            self.assertEqual(
                parser_kwargs["outputs_to_match"][2],
                r"['\"]model['\"]\s*:\s*['\"]model_1['\"]\s*,\s*['\"]rank['\"]\s*:\s*2\b",
            )
            self.assertEqual(annotator["completions_kwargs"]["max_new_tokens"], 512)
            self.assertEqual(annotator["completions_kwargs"]["thinking_token_budget"], 0)
            self.assertEqual(
                annotator["completions_kwargs"]["model_kwargs"]["reasoning_config"],
                {
                    "reasoning_start_str": "<think>",
                    "reasoning_end_str": "</think>",
                },
            )

    def test_judge_thinking_can_be_enabled_explicitly(self):
        from src.utils.alpaca_eval_runner import build_annotators_config

        cfg = AlpacaEvalConfig(
            judge_backend="vllm",
            model_name="judge-model",
            judge_enable_thinking=True,
        )
        suite = self._suite(cfg)

        with tempfile.TemporaryDirectory() as tmp:
            with patch("src.utils.alpaca_eval_runner._alpaca_eval_package_root", return_value=Path(tmp)):
                with patch("src.utils.alpaca_eval_runner.torch.cuda.device_count", return_value=1):
                    config_dir = build_annotators_config(suite, Path(tmp) / "eval")
            config = yaml.safe_load((config_dir / "configs.yaml").read_text())
            completions_kwargs = config["fdp_alpaca_eval_local_vllm"]["completions_kwargs"]

            self.assertTrue(completions_kwargs["enable_thinking"])
            self.assertEqual(completions_kwargs["thinking_token_budget"], 512)

    def test_vllm_judge_parser_extracts_rank_after_reasoning_tags(self):
        from alpaca_eval.completion_parsers import regex_parser

        completion = """</think>

</think>

[
    {'model': 'model_2', 'rank': 1},
    {'model': 'model_1', 'rank': 2}
]"""
        outputs_to_match = {
            1: r"['\"]model['\"]\s*:\s*['\"]model_1['\"]\s*,\s*['\"]rank['\"]\s*:\s*1\b",
            2: r"['\"]model['\"]\s*:\s*['\"]model_1['\"]\s*,\s*['\"]rank['\"]\s*:\s*2\b",
        }

        self.assertEqual(regex_parser(completion, outputs_to_match), [2])

    def test_cache_signature_invalidates_stale_outputs(self):
        cfg = AlpacaEvalConfig(candidate_backend="vllm")
        suite = self._suite(cfg)

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            stale_output = output_dir / "model_outputs.json"
            stale_output.write_text("[]")

            _prepare_eval_cache(suite, output_dir, "model-a", "candidate", "base")
            self.assertFalse(stale_output.exists())

            signature = json.loads((output_dir / "alpaca_eval_cache.json").read_text())["signature"]
            _prepare_eval_cache(suite, output_dir, "model-a", "candidate", "base")
            self.assertEqual(
                signature,
                json.loads((output_dir / "alpaca_eval_cache.json").read_text())["signature"],
            )


if __name__ == "__main__":
    unittest.main()
