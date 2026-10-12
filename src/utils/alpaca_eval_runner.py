from __future__ import annotations

import hashlib
import re
import json
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import torch
import yaml
from huggingface_hub import hf_hub_download

from src.schemas.suite import SuiteConfig


def _detected_tensor_parallel_size() -> int:
    """Return the number of GPUs visible to this process for vLLM tensor parallelism."""
    gpu_count = torch.cuda.device_count()
    if gpu_count < 1:
        raise RuntimeError(
            "vLLM evaluation requires at least one visible CUDA GPU; "
            "check CUDA_VISIBLE_DEVICES and the CUDA/PyTorch installation."
        )
    return gpu_count


def _alpaca_eval_package_root() -> Path:
    import alpaca_eval

    return Path(alpaca_eval.__file__).resolve().parent


def _safe_path_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "model"


def _resolve_model_config_identity(
    suite: SuiteConfig,
    model_type: str | ModelType,
    dataset_size: int | None = None,
    topk: int | None = None,
    ref_model_seed: int | None = None,
) -> tuple[str, str]:
    base_model_slug = _safe_path_component(suite.sft.cfg.base_model_id)
    kind = getattr(model_type, "value", model_type)

    if kind == "base":
        model_key = f"fdp_base_{base_model_slug}"
    elif kind == "reference":
        if ref_model_seed is None:
            raise ValueError("ref_model_seed must be provided for reference model AlpacaEval configs.")
        model_key = f"fdp_reference_seed_{ref_model_seed}_{base_model_slug}"
    elif kind == "adapter":
        if dataset_size is None or topk is None:
            raise ValueError("dataset_size and topk must be provided for adapter AlpacaEval configs.")
        diversity_tag = getattr(suite.prune.cfg, "diversity_tag", "divoff")
        model_key = f"fdp_adapter_ds{dataset_size}_topk{topk}_{diversity_tag}_{base_model_slug}"
    elif kind == "sota":
        if dataset_size is None:
            raise ValueError("dataset_size must be provided for SOTA AlpacaEval configs.")
        model_key = f"fdp_sota_ds{dataset_size}_{base_model_slug}"
    else:
        raise ValueError(f"Unsupported model_type: {model_type!r}")

    return kind, model_key


def resolve_judge_model_name(suite: SuiteConfig) -> str:
    model_name = suite.eval.cfg.alpacaeval.model_name
    if not model_name:
        raise ValueError(
            "alpacaeval.model_name must be set when alpacaeval.run is enabled."
        )
    return model_name


def resolve_eval_dataset_path(
    suite: SuiteConfig,
    eval_output_dir: Path,
    max_instances: int | None = None,
) -> Path:
    dataset_name = suite.eval.cfg.alpacaeval.dataset
    if dataset_name not in {"alpaca_eval", "alpaca_eval_gpt4_baseline"}:
        raise ValueError(
            "alpacaeval.dataset must be either 'alpaca_eval' or 'alpaca_eval_gpt4_baseline'."
        )

    filename = f"{dataset_name}.json"
    cache_path = Path(
        hf_hub_download(
            repo_id="tatsu-lab/alpaca_eval",
            repo_type="dataset",
            filename=filename,
        )
    )

    suffix = f"_max{max_instances}" if max_instances is not None else ""
    local_dataset_path = (eval_output_dir / f"alpaca_eval_dataset{suffix}.json").resolve()
    local_dataset_path.parent.mkdir(parents=True, exist_ok=True)
    if not local_dataset_path.exists():
        df_dataset = pd.read_json(cache_path)
        if max_instances is not None:
            df_dataset = df_dataset.iloc[:max_instances].copy()
        df_dataset.to_json(local_dataset_path, orient="records", indent=2)
    return local_dataset_path


def build_model_configs(
    suite: SuiteConfig,
    model_name: str,
    prompt_template_path: Path,
    model_path: str,
    model_configs_dir: Path,
    candidate_backend: str = "huggingface",
) -> Path:
    model_dir = model_configs_dir.resolve()
    model_dir.mkdir(parents=True, exist_ok=True)

    config_path = model_dir / "configs.yaml"
    prompt_path = model_dir / "prompt.txt"

    prompt_contents = prompt_template_path.read_text()
    if not prompt_path.exists() or prompt_path.read_text() != prompt_contents:
        prompt_path.write_text(prompt_contents)

    alpaca_cfg = suite.eval.cfg.alpacaeval
    fn_completions = (
        "vllm_local_completions" if candidate_backend == "vllm" else "huggingface_local_completions"
    )
    candidate_kwargs = dict(alpaca_cfg.candidate_model_kwargs)
    completions_kwargs = {
        "model_name": model_path,
        "max_new_tokens": alpaca_cfg.max_new_tokens,
        "temperature": 0.0,
        "top_p": alpaca_cfg.top_p,
        # AlpacaEval batches prompts at the template layer; our generation prompt is single-example.
        "batch_size": 1,
    }
    if candidate_backend == "vllm":
        candidate_kwargs.setdefault("dtype", alpaca_cfg.candidate_dtype)
        candidate_kwargs["tensor_parallel_size"] = _detected_tensor_parallel_size()
        candidate_kwargs.setdefault("trust_remote_code", True)
        completions_kwargs["model_kwargs"] = candidate_kwargs
    else:
        default_kwargs = {
            "torch_dtype": alpaca_cfg.candidate_dtype,
            "trust_remote_code": True,
        }
        default_kwargs.update(candidate_kwargs)
        completions_kwargs["model_kwargs"] = default_kwargs

    model_cfg = {
        model_name: {
            "prompt_template": "prompt.txt",
            "fn_completions": fn_completions,
            "completions_kwargs": completions_kwargs,
        }
    }

    desired_config = yaml.safe_dump(model_cfg, sort_keys=False)
    if not config_path.exists() or config_path.read_text() != desired_config:
        config_path.write_text(desired_config)
    return model_dir


def _clear_stale_annotator_cache(annotator_dir: Path) -> None:
    for cache_file in annotator_dir.glob("annotations_seed*_configs.json"):
        cache_file.unlink(missing_ok=True)


def build_annotators_config(
    suite: SuiteConfig,
    eval_output_dir: Path,
) -> Path:
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    judge_backend = suite.eval.cfg.alpacaeval.effective_judge_backend
    annotator_dir = (eval_output_dir / f"alpaca_eval_{judge_backend}").resolve()
    annotator_dir.mkdir(parents=True, exist_ok=True)

    package_root = _alpaca_eval_package_root()
    judge_model_name = resolve_judge_model_name(suite)

    if judge_backend == "api":
        prompt_template = (
            package_root / "evaluators_configs" / "alpaca_eval_gpt4" / "alpaca_eval.txt"
        )
        annotator_name = "fdp_alpaca_eval_api"
        annotator_cfg: Dict[str, Any] = {
            annotator_name: {
                "prompt_template": prompt_template.as_posix(),
                "fn_completions": "openai_completions",
                "completions_kwargs": {
                    "model_name": judge_model_name,
                    "max_completion_tokens": suite.eval.cfg.alpacaeval.judge_max_tokens,
                    "temperature": suite.eval.cfg.alpacaeval.temperature,
                },
                "fn_completion_parser": "ranking_parser",
                "batch_size": 1,
            }
        }
    elif judge_backend == "vllm":
        prompt_template = (
            package_root
            / "evaluators_configs"
            / "alpaca_eval_vllm_llama3_70b_fn"
            / "alpaca_eval_fn.txt"
        )
        annotator_name = "fdp_alpaca_eval_local_vllm"
        model_kwargs = dict(suite.eval.cfg.alpacaeval.effective_judge_model_kwargs)
        model_kwargs.setdefault("dtype", suite.eval.cfg.alpacaeval.judge_dtype)
        model_kwargs["tensor_parallel_size"] = _detected_tensor_parallel_size()
        model_kwargs.setdefault("trust_remote_code", True)
        model_kwargs.setdefault(
            "reasoning_config",
            {
                "reasoning_start_str": "<think>",
                "reasoning_end_str": "</think>",
            },
        )
        _clear_stale_annotator_cache(annotator_dir)
        annotator_cfg = {
            annotator_name: {
                "prompt_template": prompt_template.as_posix(),
                "fn_completions": "vllm_local_completions",
                "completions_kwargs": {
                    "model_name": judge_model_name,
                    "max_new_tokens": suite.eval.cfg.alpacaeval.judge_max_tokens,
                    "is_chatml_prompt": True,
                    # Keep the local judge batch size small so vLLM does not over-reserve KV cache.
                    "batch_size": 1,
                    "model_kwargs": model_kwargs,
                    "temperature": suite.eval.cfg.alpacaeval.temperature,
                    "top_p": suite.eval.cfg.alpacaeval.top_p,
                    "enable_thinking": suite.eval.cfg.alpacaeval.judge_enable_thinking,
                    # Keep reasoning disabled at the generation layer as a safeguard.
                    "thinking_token_budget": 0
                    if not suite.eval.cfg.alpacaeval.judge_enable_thinking
                    else suite.eval.cfg.alpacaeval.judge_max_tokens,
                },
                # Qwen reasoning models can leave <think> markers in the returned text.
                # Match model_1's rank directly so those markers cannot break parsing.
                "fn_completion_parser": "regex_parser",
                "completion_parser_kwargs": {
                    "outputs_to_match": {
                        1: r"['\"]model['\"]\s*:\s*['\"]model_1['\"]\s*,\s*['\"]rank['\"]\s*:\s*1\b",
                        2: r"['\"]model['\"]\s*:\s*['\"]model_1['\"]\s*,\s*['\"]rank['\"]\s*:\s*2\b",
                    }
                },
                # AlpacaEval's judge prompt template is single-example; batching here
                # would require a template with repeated placeholders.
                "batch_size": 1,
            }
        }
    else:
        raise ValueError(
            f"Unsupported judge_backend: {judge_backend!r}. Use 'api' or 'vllm'."
        )

    config_path = annotator_dir / "configs.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(annotator_cfg, f, sort_keys=False)
    return annotator_dir


def _build_evaluate_from_model_cmd(
    suite: SuiteConfig,
    model_configs_dir: Path,
    benchmark_path: Path,
    annotators_config: Path,
    eval_output_dir: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "src.utils.alpaca_eval_entrypoint",
        "evaluate_from_model",
        "--model_configs",
        str((model_configs_dir / "configs.yaml").resolve()),
        "--evaluation_dataset",
        str(benchmark_path.resolve()),
        "--annotators_config",
        str(annotators_config.resolve()),
        "--output_path",
        str(eval_output_dir.resolve()),
        "--is_strip_output",
        "True",
        "--is_load_outputs",
        "True",
        "--chunksize",
        str(suite.eval.params.batch_size),
        "--base_dir",
        str(model_configs_dir.resolve()),
    ]

    if suite.eval.cfg.alpacaeval.max_instances is not None:
        cmd += ["--max_instances", str(suite.eval.cfg.alpacaeval.max_instances)]

    return cmd


def _run_evaluate_from_model_subprocess(
    suite: SuiteConfig,
    model_configs_dir: Path,
    benchmark_path: Path,
    annotators_config: Path,
    eval_output_dir: Path,
) -> None:
    cmd = _build_evaluate_from_model_cmd(
        suite=suite,
        model_configs_dir=model_configs_dir,
        benchmark_path=benchmark_path,
        annotators_config=annotators_config,
        eval_output_dir=eval_output_dir,
    )
    printable_cmd = " ".join(shlex.quote(part) for part in cmd)
    max_instances = suite.eval.cfg.alpacaeval.max_instances
    print(
        "Running AlpacaEval subprocess "
        f"(max_instances={max_instances}, dataset_path={benchmark_path.resolve()}):\n{printable_cmd}"
    )

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("Python could not be found to launch AlpacaEval.") from exc
    except subprocess.CalledProcessError:
        raise RuntimeError("AlpacaEval subprocess failed. See the command output above for details.")


def _remove_stale_model_outputs_cache(
    eval_output_dir: Path,
    expected_generator: str,
    expected_max_instances: int | None = None,
) -> None:
    model_outputs_path = eval_output_dir / "model_outputs.json"
    if not model_outputs_path.exists():
        return

    try:
        with open(model_outputs_path, "r") as f:
            cached_outputs = json.load(f)
    except (json.JSONDecodeError, OSError):
        model_outputs_path.unlink(missing_ok=True)
        return

    if isinstance(cached_outputs, list):
        if expected_max_instances is not None and len(cached_outputs) != expected_max_instances:
            model_outputs_path.unlink(missing_ok=True)
            return
        if cached_outputs:
            current_generator = cached_outputs[0].get("generator")
            if current_generator != expected_generator:
                model_outputs_path.unlink(missing_ok=True)


def _clear_stale_eval_artifacts(eval_output_dir: Path) -> None:
    for filename in (
        "model_outputs.json",
        "reference_outputs.json",
        "leaderboard.csv",
        "annotations.json",
        "annotation.json",
    ):
        (eval_output_dir / filename).unlink(missing_ok=True)


def _filter_alpaca_eval_summary(row: dict[str, Any]) -> dict[str, Any]:
    if "length_controlled_winrate" not in row:
        raise KeyError("AlpacaEval output is missing length_controlled_winrate")
    important_keys = [
        "alpacaeval",
        "n_total",
        "n_wins",
        "n_wins_base",
        "n_draws",
        "standard_error",
        "judge_backend",
        "judge_model_name",
        "alpaca_eval_dataset",
    ]
    filtered = {k: row[k] for k in important_keys if k in row}
    filtered["alpacaeval"] = float(row["length_controlled_winrate"])
    return filtered


def _load_alpaca_eval_outputs(eval_output_dir: Path) -> tuple[dict[str, Any], Dict[str, Any]]:
    leaderboard_path = eval_output_dir / "leaderboard.csv"
    annotations_path = eval_output_dir / "annotations.json"

    if not leaderboard_path.exists():
        raise FileNotFoundError(
            f"Expected AlpacaEval leaderboard at {leaderboard_path}, but it was not created."
        )

    leaderboard_df = pd.read_csv(leaderboard_path, index_col=0)
    if leaderboard_df.empty:
        raise ValueError(f"AlpacaEval leaderboard at {leaderboard_path} is empty.")

    annotations: Dict[str, Any] | list[dict[str, Any]]
    if annotations_path.exists():
        with open(annotations_path, "r") as f:
            annotations = json.load(f)
    else:
        annotations = []

    leader_row = leaderboard_df.iloc[0].to_dict()
    return leader_row, {
        "leaderboard": leaderboard_df.to_dict(orient="records"),
        "annotations": annotations,
    }


def _cache_signature(
    suite: SuiteConfig,
    model_path: str,
    model_name: str,
    model_type: str | ModelType,
) -> str:
    kind = getattr(model_type, "value", model_type)
    payload = {
        "model_path": model_path,
        "model_name": model_name,
        "model_type": kind,
        "candidate_backend": suite.eval.cfg.alpacaeval.effective_candidate_backend,
        "candidate_dtype": suite.eval.cfg.alpacaeval.candidate_dtype,
        "candidate_model_kwargs": suite.eval.cfg.alpacaeval.candidate_model_kwargs,
        "max_new_tokens": suite.eval.cfg.alpacaeval.max_new_tokens,
        "top_p": suite.eval.cfg.alpacaeval.top_p,
        "dataset": suite.eval.cfg.alpacaeval.dataset,
        "max_instances": suite.eval.cfg.alpacaeval.max_instances,
        "judge_backend": suite.eval.cfg.alpacaeval.effective_judge_backend,
        "judge_dtype": suite.eval.cfg.alpacaeval.judge_dtype,
        "judge_model_name": suite.eval.cfg.alpacaeval.model_name,
        "judge_model_kwargs": suite.eval.cfg.alpacaeval.effective_judge_model_kwargs,
        "judge_max_tokens": suite.eval.cfg.alpacaeval.judge_max_tokens,
        "judge_enable_thinking": suite.eval.cfg.alpacaeval.judge_enable_thinking,
        "temperature": suite.eval.cfg.alpacaeval.temperature,
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _prepare_eval_cache(
    suite: SuiteConfig,
    eval_output_dir: Path,
    model_path: str,
    model_name: str,
    model_type: str | ModelType,
) -> None:
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = eval_output_dir / "alpaca_eval_cache.json"
    signature = _cache_signature(suite, model_path, model_name, model_type)
    existing_signature = None
    if cache_path.exists():
        try:
            with open(cache_path, "r") as f:
                existing_signature = json.load(f).get("signature")
        except (json.JSONDecodeError, OSError):
            pass
    if existing_signature != signature:
        _clear_stale_eval_artifacts(eval_output_dir)
        with open(cache_path, "w") as f:
            json.dump({"signature": signature}, f, indent=2)


def run_alpaca_eval_from_model(
    suite: SuiteConfig,
    model_path: str,
    eval_output_dir: Path,
    model_type: str | ModelType,
    dataset_size: int | None = None,
    topk: int | None = None,
    ref_model_seed: int | None = None,
) -> tuple[dict[str, Any], Dict[str, Any]]:
    benchmark_path = resolve_eval_dataset_path(
        suite,
        eval_output_dir,
        max_instances=suite.eval.cfg.alpacaeval.max_instances,
    )
    judge_model_name = resolve_judge_model_name(suite)

    model_prompt_path = Path(__file__).resolve().parents[2] / "_alpaca_eval_prompt.txt"
    if not model_prompt_path.exists():
        model_prompt_path.write_text("{instruction}\n")

    _, model_name = _resolve_model_config_identity(
        suite=suite,
        model_type=model_type,
        dataset_size=dataset_size,
        topk=topk,
        ref_model_seed=ref_model_seed,
    )

    with tempfile.TemporaryDirectory(prefix="fdp_alpaca_eval_") as temp_dir:
        temp_root = Path(temp_dir)
        model_configs_dir = temp_root / "model_configs"
        annotators_config_dir = temp_root / "annotators"

        print("\n" + "=" * 60)
        print("PHASE 4A: Generating model outputs with AlpacaEval evaluate_from_model...")
        print("=" * 60)

        model_configs_dir = build_model_configs(
            suite=suite,
            model_name=model_name,
            prompt_template_path=model_prompt_path,
            model_path=model_path,
            model_configs_dir=model_configs_dir,
            candidate_backend=suite.eval.cfg.alpacaeval.effective_candidate_backend,
        )
        if suite.eval.cfg.alpacaeval.max_instances is not None:
            _clear_stale_eval_artifacts(eval_output_dir=eval_output_dir)
        _prepare_eval_cache(
            suite=suite,
            eval_output_dir=eval_output_dir,
            model_path=model_path,
            model_name=model_name,
            model_type=model_type,
        )
        _remove_stale_model_outputs_cache(
            eval_output_dir=eval_output_dir,
            expected_generator=model_name,
            expected_max_instances=suite.eval.cfg.alpacaeval.max_instances,
        )

        print("\n" + "=" * 60)
        print("PHASE 4B: Building AlpacaEval judge config and scoring outputs...")
        print("=" * 60)

        annotators_config = build_annotators_config(suite=suite, eval_output_dir=annotators_config_dir)
        if suite.eval.cfg.alpacaeval.effective_judge_backend == "api":
            import alpaca_eval.main as alpaca_eval_main

            max_instances = suite.eval.cfg.alpacaeval.max_instances
            print(
                "Running AlpacaEval directly in-process for api backend "
                f"(max_instances={max_instances}, dataset_path={benchmark_path.resolve()})"
            )
            alpaca_eval_main.evaluate_from_model(
                model_configs=(model_configs_dir / "configs.yaml").resolve(),
                evaluation_dataset=benchmark_path.resolve(),
                annotators_config=annotators_config.resolve(),
                output_path=eval_output_dir.resolve(),
                max_instances=max_instances,
                is_strip_output=True,
                is_load_outputs=True,
                chunksize=None,
                base_dir=str(model_configs_dir.resolve()),
            )
        else:
            _run_evaluate_from_model_subprocess(
                suite=suite,
                model_configs_dir=model_configs_dir,
                benchmark_path=benchmark_path,
                annotators_config=annotators_config,
                eval_output_dir=eval_output_dir,
            )
    leader_row, raw = _load_alpaca_eval_outputs(eval_output_dir)

    leader_row["judge_backend"] = suite.eval.cfg.alpacaeval.effective_judge_backend
    leader_row["judge_model_name"] = judge_model_name
    leader_row["alpaca_eval_dataset"] = suite.eval.cfg.alpacaeval.dataset
    return _filter_alpaca_eval_summary(leader_row), raw
