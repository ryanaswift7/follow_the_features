"""Run directly comparable GRAPE and canonical single-proxy S2L experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import multiprocessing as mp
import os
import re
import shutil
from pathlib import Path

import torch
import yaml
from datasets import Dataset, load_dataset
from peft import get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
)
from trl.trainer.sft_trainer import SFTTrainer

from src.prune import get_formatting_function
from src.schemas.sota import SOTAConfig
from src.schemas.suite import SuiteConfig
from src.sft import PEFT_CONFIG, generate_sft_trainer_args
from src.sota_methods.grape import format_grape_prompt, select_grape_responses
from src.sota_methods.s2l import collect_checkpoint_trajectories, select_s2l
from src.utils.enums import ModelType
from src.utils.evals import eval_and_save
from src.utils.load_model_and_tokenizer import load_model_and_tokenizer


logger = logging.getLogger(__name__)


def _atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _atomic_save_dataset(dataset: Dataset, path: Path, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    dataset.save_to_disk(temporary)
    _atomic_json(temporary / "_sota_complete.json", metadata)
    if path.exists():
        shutil.rmtree(path)
    os.replace(temporary, path)


def _load_complete_dataset(path: Path) -> Dataset | None:
    marker = path / "_sota_complete.json"
    if not path.is_dir() or not marker.is_file():
        return None
    try:
        return Dataset.load_from_disk(path)
    except Exception as exc:
        logger.warning("Ignoring incomplete dataset artifact %s: %s", path, exc)
        return None


def _latest_checkpoint(root: Path, before_step: int | None = None) -> Path | None:
    checkpoints = []
    for path in root.glob("checkpoint-*"):
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        if match and (before_step is None or int(match.group(1)) < before_step):
            if (path / "trainer_state.json").is_file():
                checkpoints.append((int(match.group(1)), path))
    return max(checkpoints, default=(0, None), key=lambda item: item[0])[1]


def _suite_hash_payload(suite: SuiteConfig) -> dict:
    return {
        "title": suite.title,
        "seeds": suite.seeds,
        "debug_mode": suite.debug_mode,
        "full_data_reference": suite.full_data_reference,
        "prune": suite.prune.model_dump(mode="json"),
        "sft": suite.sft.model_dump(mode="json"),
        "eval": suite.eval.model_dump(mode="json"),
    }


def _config_hashes(suite: SuiteConfig, sota: SOTAConfig) -> tuple[str, str, str]:
    suite_payload = _suite_hash_payload(suite)
    sota_payload = sota.model_dump(mode="json")
    suite_hash = hashlib.sha256(
        json.dumps(suite_payload, sort_keys=True).encode()
    ).hexdigest()
    sota_hash = hashlib.sha256(
        json.dumps(sota_payload, sort_keys=True).encode()
    ).hexdigest()
    run_hash = hashlib.sha256(f"{suite_hash}:{sota_hash}".encode()).hexdigest()
    return suite_hash, sota_hash, run_hash


def _write_config_copy(
    destination: Path,
    source: Path | None,
    payload: dict,
) -> None:
    if source is not None and source.is_file():
        temporary = destination.with_name(f".{destination.name}.tmp")
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
        return
    temporary = destination.with_name(f".{destination.name}.tmp")
    with temporary.open("w") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    os.replace(temporary, destination)


def _prepare_run_root(
    base_output_root: Path,
    suite: SuiteConfig,
    sota: SOTAConfig,
    suite_config_path: Path | None,
    sota_config_path: Path | None,
) -> Path:
    suite_hash, sota_hash, run_hash = _config_hashes(suite, sota)
    output_root = base_output_root / f"config-{run_hash[:16]}"
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "run_manifest.json"
    manifest = {
        "run_hash": run_hash,
        "suite_hash": suite_hash,
        "sota_hash": sota_hash,
        "suite_title": suite.title,
        "dataset_id": sota.dataset_id,
    }
    if manifest_path.exists():
        with manifest_path.open() as handle:
            existing = json.load(handle)
        if existing.get("run_hash") != run_hash:
            raise ValueError(f"Hash collision detected in SOTA output directory {output_root}")
        logger.info("Resuming compatible SOTA run from %s", output_root)
    else:
        _atomic_json(manifest_path, manifest)
        logger.info("Initialized SOTA run manifest: %s", manifest_path)

    suite_copy = output_root / "suite_config.yaml"
    sota_copy = output_root / "sota_config.yaml"
    if not suite_copy.exists():
        _write_config_copy(suite_copy, suite_config_path, _suite_hash_payload(suite))
    if not sota_copy.exists():
        _write_config_copy(sota_copy, sota_config_path, sota.model_dump(mode="json"))
    return output_root


def _generate_teacher_worker(
    teacher_id: str,
    prompts_path: Path,
    responses_path: Path,
    complete_path: Path,
    max_tokens: int,
    temperature: float,
    seed: int,
) -> None:
    """Generate one teacher's responses in an isolated process."""
    try:
        from vllm import LLM, SamplingParams

        with prompts_path.open() as handle:
            payload = json.load(handle)
        prompts = payload["prompts"]
        logger.info(
            "GRAPE worker loading teacher=%s, prompts=%s, seed=%s",
            teacher_id,
            len(prompts),
            seed,
        )
        llm = LLM(model=teacher_id)
        outputs = llm.generate(
            prompts,
            SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
            ),
            use_tqdm=True,
        )
        responses = [output.outputs[0].text for output in outputs]
        if len(responses) != len(prompts):
            raise RuntimeError(
                f"Teacher {teacher_id} returned {len(responses)} responses for "
                f"{len(prompts)} prompts"
            )
        _atomic_json(
            responses_path,
            {"teacher_id": teacher_id, "seed": seed, "responses": responses},
        )
        _atomic_json(
            complete_path,
            {"teacher_id": teacher_id, "seed": seed, "responses": len(responses)},
        )
    except Exception as exc:
        _atomic_json(
            complete_path.with_name("worker_error.json"),
            {"teacher_id": teacher_id, "seed": seed, "error": repr(exc)},
        )
        raise


class _CheckpointCallback(TrainerCallback):
    def __init__(self, root: Path, checkpoints: list[int]):
        self.root = root
        self.checkpoints = set(checkpoints)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step in self.checkpoints:
            path = self.root / f"step-{state.global_step}"
            path.mkdir(parents=True, exist_ok=True)
            torch.save(kwargs["model"].state_dict(), path / "model_state.pt")
            logger.info("Saved S2L proxy checkpoint: step=%s, path=%s", state.global_step, path)
        return control


def _messages_dataset(dataset, dataset_id, tokenizer):
    logger.info("Formatting %s dataset rows as chat messages", len(dataset))
    formatter = get_formatting_function(dataset_id)
    sources = dataset["source"] if "source" in dataset.column_names else None
    formatted = dataset.map(
        lambda batch: formatter(tokenizer, batch, is_inference=False),
        batched=True,
        remove_columns=dataset.column_names,
    )
    if sources is not None:
        formatted = formatted.add_column("source", sources)
    logger.info("Formatted %s dataset rows", len(formatted))
    return formatted


def _train_proxy(dataset, suite, sota, root):
    checkpoints = sota.s2l.trajectory_checkpoints
    complete_marker = root / "_sota_complete.json"
    checkpoint_states = [root / f"step-{step}" / "model_state.pt" for step in checkpoints]
    if complete_marker.is_file() and all(path.is_file() for path in checkpoint_states):
        logger.info("S2L proxy training already complete; reusing %s", root)
        return
    logger.info(
        "S2L proxy training started: model=%s, rows=%s, max_steps=%s, checkpoints=%s",
        sota.s2l.proxy_model_id,
        len(dataset),
        max(checkpoints),
        checkpoints,
    )
    model, tokenizer = load_model_and_tokenizer(
        sota.s2l.proxy_model_id, sota.s2l.proxy_tokenizer_id, is_inference=False
    )

    root.mkdir(parents=True, exist_ok=True)
    args = generate_sft_trainer_args(
        suite=suite,
        output_dir=root / "trainer",
        num_training_examples=len(dataset),
    )
    args.max_steps = max(sota.s2l.trajectory_checkpoints)
    args.save_strategy = "steps"
    args.save_steps = 100
    args.save_total_limit = None
    args.disable_tqdm = False
    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": dataset,
        "processing_class": tokenizer,
        "callbacks": [_CheckpointCallback(root, sota.s2l.trajectory_checkpoints)],
    }
    if not suite.sft.cfg.full_ft:
        trainer_kwargs["peft_config"] = PEFT_CONFIG
    trainer = SFTTrainer(**trainer_kwargs)
    resume_checkpoint = _latest_checkpoint(root / "trainer", before_step=args.max_steps)
    if resume_checkpoint is not None:
        logger.info("Resuming S2L proxy training from %s", resume_checkpoint)
        trainer.train(resume_from_checkpoint=str(resume_checkpoint))
    else:
        trainer.train()
    if not all(path.is_file() for path in checkpoint_states):
        missing = [str(path) for path in checkpoint_states if not path.is_file()]
        raise RuntimeError(f"S2L proxy training finished without required checkpoints: {missing}")
    _atomic_json(
        complete_marker,
        {"max_steps": max(checkpoints), "trajectory_checkpoints": checkpoints, "rows": len(dataset)},
    )
    logger.info("S2L proxy training complete; checkpoints saved under %s", root)
    return model, tokenizer


def _generate_candidates(
    dataset,
    teacher_ids,
    tokenizer,
    max_tokens,
    temperature,
    seed,
    teacher_root: Path,
):
    grouped = {}
    original_response_count = 0
    for row in dataset:
        messages = row["messages"]
        instruction = next(message["content"] for message in messages if message["role"] == "user")
        assistants = [message["content"] for message in messages if message["role"] == "assistant"]
        entry = grouped.setdefault(instruction, {"instruction": instruction, "responses": [], "sources": []})
        for response in assistants:
            entry["responses"].append(response)
            entry["sources"].append("original")
            original_response_count += 1

    logger.info(
        "GRAPE candidate pool initialized: rows=%s, unique_instructions=%s, "
        "original_responses=%s, teachers=%s",
        len(dataset),
        len(grouped),
        original_response_count,
        len(teacher_ids),
    )
    if not teacher_ids:
        logger.info("No GRAPE teachers configured; using original dataset responses only")

    prompts = [
        format_grape_prompt(tokenizer, instruction)
        for instruction in grouped
    ]
    teacher_root.mkdir(parents=True, exist_ok=True)
    context = mp.get_context("spawn")
    for teacher_id in teacher_ids:
        teacher_slug = re.sub(r"[^A-Za-z0-9._-]+", "_", teacher_id).strip("_")
        teacher_dir = teacher_root / teacher_slug
        teacher_dir.mkdir(parents=True, exist_ok=True)
        prompts_path = teacher_dir / "prompts.json"
        responses_path = teacher_dir / "responses.json"
        complete_path = teacher_dir / "complete.json"
        if complete_path.is_file() and responses_path.is_file():
            with responses_path.open() as handle:
                cached = json.load(handle)
            responses = cached.get("responses", [])
            if (
                cached.get("teacher_id") == teacher_id
                and cached.get("seed") == seed
                and len(responses) == len(prompts)
            ):
                logger.info("Reusing GRAPE teacher output: teacher=%s", teacher_id)
                for instruction, response in zip(grouped, responses):
                    grouped[instruction]["responses"].append(response)
                    grouped[instruction]["sources"].append(teacher_id)
                continue

        _atomic_json(
            prompts_path,
            {"teacher_id": teacher_id, "seed": seed, "prompts": prompts},
        )
        logger.info(
            "GRAPE teacher subprocess started: teacher=%s, prompts=%s, max_tokens=%s, temperature=%s",
            teacher_id,
            len(prompts),
            max_tokens,
            temperature,
        )
        worker = context.Process(
            target=_generate_teacher_worker,
            args=(
                teacher_id,
                prompts_path,
                responses_path,
                complete_path,
                max_tokens,
                temperature,
                seed,
            ),
        )
        worker.start()
        worker.join()
        if worker.exitcode != 0:
            error_path = teacher_dir / "worker_error.json"
            raise RuntimeError(
                f"GRAPE teacher subprocess failed for {teacher_id} "
                f"(exit code {worker.exitcode}); see {error_path}"
            )
        with responses_path.open() as handle:
            generated = json.load(handle)
        responses = generated.get("responses", [])
        if (
            generated.get("teacher_id") != teacher_id
            or generated.get("seed") != seed
            or len(responses) != len(prompts)
            or not complete_path.is_file()
        ):
            raise RuntimeError(f"Incomplete GRAPE teacher artifact for {teacher_id}: {teacher_dir}")
        for instruction, response in zip(grouped, responses):
            grouped[instruction]["responses"].append(response)
            grouped[instruction]["sources"].append(teacher_id)
        logger.info("GRAPE teacher subprocess complete: teacher=%s", teacher_id)
    logger.info("GRAPE candidate generation complete")
    return list(grouped.values())


def _train_and_evaluate(dataset, suite, method, budget, output_root):
    model_dir = output_root / "models" / method / f"budget-{budget}"
    model_dir.mkdir(parents=True, exist_ok=True)
    complete_marker = model_dir / "_sota_complete.json"
    if complete_marker.is_file():
        logger.info("%s model already complete; reusing %s", method.upper(), model_dir)
        model, tokenizer = load_model_and_tokenizer(
            model_id=model_dir, tokenizer_id=model_dir, is_inference=False
        )
        return model, tokenizer, model_dir
    logger.info(
        "%s training started: budget=%s, rows=%s, output=%s",
        method.upper(),
        budget,
        len(dataset),
        model_dir,
    )
    model, tokenizer = load_model_and_tokenizer(
        suite.sft.cfg.base_model_id, suite.sft.cfg.tokenizer_id, is_inference=False
    )
    args = generate_sft_trainer_args(
        suite=suite,
        output_dir=model_dir / "trainer",
        num_training_examples=len(dataset),
    )
    args.save_strategy = "steps"
    args.save_steps = 100
    args.save_total_limit = None
    args.disable_tqdm = False
    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": dataset,
        "processing_class": tokenizer,
    }
    if not suite.sft.cfg.full_ft:
        trainer_kwargs["peft_config"] = PEFT_CONFIG
    trainer = SFTTrainer(**trainer_kwargs)
    resume_checkpoint = _latest_checkpoint(model_dir / "trainer")
    if resume_checkpoint is not None:
        logger.info("Resuming %s training from %s", method.upper(), resume_checkpoint)
        trainer.train(resume_from_checkpoint=str(resume_checkpoint))
    else:
        trainer.train()
    if not suite.sft.cfg.full_ft:
        model = trainer.model.merge_and_unload()
        model.save_pretrained(model_dir)
    else:
        trainer.save_model(model_dir)
        model = trainer.model
    tokenizer.save_pretrained(model_dir)
    _atomic_json(
        complete_marker,
        {"method": method, "budget": budget, "rows": len(dataset)},
    )
    logger.info("%s training complete: model=%s", method.upper(), model_dir)
    return model, tokenizer, model_dir


def _average_values(values):
    if not values:
        return None
    if all(isinstance(value, dict) for value in values):
        keys = sorted({key for value in values for key in value})
        return {
            key: _average_values([value[key] for value in values if key in value])
            for key in keys
        }
    if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
        return sum(values) / len(values)
    return values[0]


def _upsert_result(results: list[dict], entry: dict) -> None:
    key = (entry["seed"], entry["method"], entry["budget"])
    results[:] = [
        existing
        for existing in results
        if (existing["seed"], existing["method"], existing["budget"]) != key
    ]
    results.append(entry)


def _write_comparison_files(output_root: Path, raw_results: list[dict]) -> None:
    _atomic_json(output_root / "comparison_by_seed.json", raw_results)
    grouped = {}
    for entry in raw_results:
        grouped.setdefault((entry["method"], entry["budget"]), []).append(entry)
    averaged = []
    for (method, budget), entries in sorted(grouped.items()):
        averaged.append(
            {
                "method": method,
                "budget": budget,
                "seeds": sorted(entry["seed"] for entry in entries),
                "results": _average_values([entry["results"] for entry in entries]),
            }
        )
    _atomic_json(output_root / "comparison.json", averaged)


def _run_seed(suite, sota, raw, budgets, output_root, seed, raw_results):
    seed_suite = suite.for_seed(seed)
    seed_root = output_root / "seeds" / f"seed_{seed}"
    seed_root.mkdir(parents=True, exist_ok=True)
    logger.info("Starting SOTA seed %s", seed)

    grape_rows = {}
    missing_grape = []
    for budget in budgets:
        path = seed_root / "datasets" / "grape" / f"budget-{budget}"
        selected = _load_complete_dataset(path)
        if selected is None:
            missing_grape.append(budget)
        else:
            grape_rows[budget] = selected

    if missing_grape:
        candidates_path = seed_root / "grape_candidates.json"
        if candidates_path.is_file():
            with candidates_path.open() as handle:
                candidates = json.load(handle)["candidates"]
            logger.info("Reusing GRAPE candidates for seed %s", seed)
        else:
            logger.info("Generating GRAPE candidates for seed %s", seed)
            tokenizer = AutoTokenizer.from_pretrained(seed_suite.sft.cfg.tokenizer_id)
            candidates = _generate_candidates(
                raw, sota.grape.teacher_ids, tokenizer, sota.grape.max_tokens,
                sota.grape.temperature, seed, seed_root / "grape_teachers",
            )
            _atomic_json(candidates_path, {"candidates": candidates, "rows": len(raw), "seed": seed})

        tokenizer = AutoTokenizer.from_pretrained(seed_suite.sft.cfg.tokenizer_id)
        target_model = AutoModelForCausalLM.from_pretrained(
            seed_suite.sft.cfg.base_model_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        for budget in missing_grape:
            selected = Dataset.from_list(select_grape_responses(
                target_model, tokenizer, candidates, budget, sota.grape.batch_size,
                max_length=seed_suite.sft.cfg.max_seq_length,
            ))
            _atomic_save_dataset(
                selected, seed_root / "datasets" / "grape" / f"budget-{budget}",
                {"method": "grape", "budget": budget, "seed": seed},
            )
            grape_rows[budget] = selected
        del target_model
        torch.cuda.empty_cache()

    s2l_rows = {}
    missing_s2l = []
    for budget in budgets:
        path = seed_root / "datasets" / "s2l" / f"budget-{budget}"
        selected = _load_complete_dataset(path)
        if selected is None:
            missing_s2l.append(budget)
        else:
            s2l_rows[budget] = selected

    if missing_s2l:
        proxy_root = seed_root / "proxy"
        _train_proxy(raw, seed_suite, sota, proxy_root)
        trajectory_path = seed_root / "s2l_trajectories.pt"
        metadata_path = seed_root / "s2l_trajectories.json"
        trajectories = None
        if trajectory_path.is_file() and metadata_path.is_file():
            with metadata_path.open() as handle:
                metadata = json.load(handle)
            if metadata.get("checkpoints") == sota.s2l.trajectory_checkpoints and metadata.get("rows") == len(raw):
                trajectories = torch.load(trajectory_path, weights_only=True)
        if trajectories is None:
            proxy_model, proxy_tokenizer = load_model_and_tokenizer(
                sota.s2l.proxy_model_id, sota.s2l.proxy_tokenizer_id, is_inference=False
            )
            if not seed_suite.sft.cfg.full_ft:
                proxy_model = get_peft_model(proxy_model, PEFT_CONFIG)
            trajectories = collect_checkpoint_trajectories(
                proxy_model, proxy_tokenizer, raw,
                [proxy_root / f"step-{step}" for step in sota.s2l.trajectory_checkpoints],
                seed_suite.sft.cfg.max_seq_length, seed_suite.sft.params.batch_size,
            )
            _atomic_torch_save(trajectory_path, trajectories)
            _atomic_json(metadata_path, {"checkpoints": sota.s2l.trajectory_checkpoints, "rows": len(raw), "seed": seed})
        for budget in missing_s2l:
            selected = raw.select(select_s2l(
                trajectories, budget, sota.s2l.n_clusters, seed,
                sources=raw["source"] if "source" in raw.column_names else None,
            ))
            _atomic_save_dataset(
                selected, seed_root / "datasets" / "s2l" / f"budget-{budget}",
                {"method": "s2l", "budget": budget, "seed": seed},
            )
            s2l_rows[budget] = selected

    seed_results = []
    for budget in budgets:
        for method, selected in (("grape", grape_rows[budget]), ("s2l", s2l_rows[budget])):
            model_dir = seed_root / "models" / method / f"budget-{budget}"
            result_path = model_dir / "result.json"
            if result_path.is_file():
                with result_path.open() as handle:
                    eval_result = json.load(handle)
            else:
                model, tokenizer, model_dir = _train_and_evaluate(
                    selected, seed_suite, method, budget, seed_root
                )
                eval_result, _ = eval_and_save(
                    model=model, tokenizer=tokenizer, suite=seed_suite,
                    model_type=ModelType.SOTA, dataset_size=budget, topk=None,
                    gen_cache_dir=model_dir / "generation_cache",
                    save_results_and_config=False,
                    sota_model_path=str(model_dir),
                    sota_eval_output_dir=model_dir / "alpaca_eval",
                )
                _atomic_json(result_path, eval_result)
            entry = {"seed": seed, "method": method, "budget": budget, "results": eval_result}
            _upsert_result(raw_results, entry)
            seed_results.append(entry)
            _write_comparison_files(output_root, raw_results)
    _atomic_json(seed_root / "comparison.json", seed_results)


def run(
    suite: SuiteConfig,
    sota: SOTAConfig,
    suite_config_path: Path | None = None,
    sota_config_path: Path | None = None,
):
    seeds = sota.seeds if sota.seeds is not None else suite.seeds
    invalid_seeds = [seed for seed in seeds if seed not in suite.seeds]
    if invalid_seeds:
        raise ValueError(f"SOTA seeds {invalid_seeds} are not configured in the main suite")
    budgets = suite.prune.params.dataset_sizes
    output_root = _prepare_run_root(
        base_output_root=sota.output_dir or (suite.dirs.sota_results / "grape_s2l"),
        suite=suite, sota=sota,
        suite_config_path=suite_config_path, sota_config_path=sota_config_path,
    )
    raw = load_dataset(sota.dataset_id, split=sota.split)
    raw = _messages_dataset(raw, sota.dataset_id, AutoTokenizer.from_pretrained(suite.sft.cfg.tokenizer_id))
    unique_instructions = len({
        next(message["content"] for message in row["messages"] if message["role"] == "user")
        for row in raw
    })
    oversized = [budget for budget in budgets if budget > unique_instructions]
    if oversized:
        raise ValueError(f"GRAPE budgets {oversized} exceed the {unique_instructions} unique instructions")
    logger.info("Running SOTA comparison for seeds=%s, budgets=%s, output=%s", seeds, budgets, output_root)
    raw_results = []
    for seed in seeds:
        _run_seed(suite, sota, raw, budgets, output_root, seed, raw_results)
    _write_comparison_files(output_root, raw_results)
    logger.info("GRAPE/S2L comparison complete: results=%s", output_root / "comparison.json")


def main():
    parser = argparse.ArgumentParser(description="Run direct GRAPE/S2L comparisons")
    parser.add_argument("-c", "--config", required=True, help="Methodology SuiteConfig YAML")
    parser.add_argument("-s", "--sota-config", required=True, help="Separate SOTA YAML")
    args = parser.parse_args()
    suite_config_path = Path(args.config).resolve()
    sota_config_path = Path(args.sota_config).resolve()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    run(
        SuiteConfig.from_yaml(suite_config_path),
        SOTAConfig.from_yaml(sota_config_path),
        suite_config_path=suite_config_path,
        sota_config_path=sota_config_path,
    )


if __name__ == "__main__":
    main()
