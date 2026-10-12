"""Run directly comparable GRAPE and canonical single-proxy S2L experiments."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from peft import get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from vllm import LLM, SamplingParams

from src.prune import get_formatting_function
from src.schemas.sota import SOTAConfig
from src.schemas.suite import SuiteConfig
from src.sft import PEFT_CONFIG
from src.sota_methods.grape import format_grape_prompt, select_grape_responses
from src.sota_methods.s2l import collect_checkpoint_trajectories, select_s2l
from src.utils.enums import ModelType
from src.utils.evals import eval_and_save
from src.utils.load_model_and_tokenizer import load_model_and_tokenizer


class _CheckpointCallback(TrainerCallback):
    def __init__(self, root: Path, checkpoints: list[int]):
        self.root = root
        self.checkpoints = set(checkpoints)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step in self.checkpoints:
            path = self.root / f"step-{state.global_step}"
            path.mkdir(parents=True, exist_ok=True)
            torch.save(kwargs["model"].state_dict(), path / "model_state.pt")
        return control


class _Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples):
        width = max(len(example["input_ids"]) for example in examples)
        pad = self.tokenizer.pad_token_id or 0
        ids, labels, masks = [], [], []
        for example in examples:
            padding = width - len(example["input_ids"])
            ids.append(example["input_ids"] + [pad] * padding)
            labels.append(example["labels"] + [-100] * padding)
            masks.append([1] * (width - padding) + [0] * padding)
        return {
            "input_ids": torch.tensor(ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(masks),
        }


def _messages_dataset(dataset, dataset_id, tokenizer):
    formatter = get_formatting_function(dataset_id)
    sources = dataset["source"] if "source" in dataset.column_names else None
    formatted = dataset.map(
        lambda batch: formatter(tokenizer, batch, is_inference=False),
        batched=True,
        remove_columns=dataset.column_names,
    )
    if sources is not None:
        formatted = formatted.add_column("source", sources)
    return formatted


def _tokenize_messages(dataset, tokenizer, max_length):
    def encode(example):
        messages = example["messages"]
        full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
        ids = tokenizer(full, add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]
        prompt_ids = tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]
        labels = list(ids)
        labels[: min(len(prompt_ids), len(labels))] = [-100] * min(len(prompt_ids), len(labels))
        return {"input_ids": ids, "labels": labels}

    return dataset.map(encode, remove_columns=dataset.column_names)


def _train_proxy(dataset, suite, sota, root):
    model, tokenizer = load_model_and_tokenizer(
        sota.s2l.proxy_model_id, sota.s2l.proxy_tokenizer_id, is_inference=False
    )
    tokenized = _tokenize_messages(dataset, tokenizer, suite.sft.cfg.max_seq_length)
    if not suite.sft.cfg.full_ft:
        model = get_peft_model(model, PEFT_CONFIG)

    root.mkdir(parents=True, exist_ok=True)
    use_bf16 = torch.cuda.is_available()
    args = TrainingArguments(
        output_dir=str(root / "trainer"),
        max_steps=max(sota.s2l.trajectory_checkpoints),
        per_device_train_batch_size=suite.sft.params.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=suite.sft.cfg.learning_rate,
        lr_scheduler_type=suite.sft.cfg.lr_scheduler_type,
        warmup_steps=0,
        weight_decay=0.01,
        bf16=use_bf16,
        fp16=False,
        save_strategy="no",
        report_to="none",
        seed=suite.seed,
        data_seed=suite.seed,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=_Collator(tokenizer),
        callbacks=[_CheckpointCallback(root, sota.s2l.trajectory_checkpoints)],
    )
    trainer.train()
    return model, tokenizer


def _generate_candidates(dataset, teacher_ids, tokenizer, max_tokens, temperature):
    grouped = {}
    for row in dataset:
        messages = row["messages"]
        instruction = next(message["content"] for message in messages if message["role"] == "user")
        assistants = [message["content"] for message in messages if message["role"] == "assistant"]
        entry = grouped.setdefault(instruction, {"instruction": instruction, "responses": [], "sources": []})
        for response in assistants:
            entry["responses"].append(response)
            entry["sources"].append("original")

    prompts = [
        format_grape_prompt(tokenizer, instruction)
        for instruction in grouped
    ]
    for teacher_id in teacher_ids:
        llm = LLM(model=teacher_id)
        outputs = llm.generate(
            prompts,
            SamplingParams(temperature=temperature, max_tokens=max_tokens),
        )
        for instruction, output in zip(grouped, outputs):
            grouped[instruction]["responses"].append(output.outputs[0].text)
            grouped[instruction]["sources"].append(teacher_id)
        del llm
    return list(grouped.values())


def _train_and_evaluate(dataset, suite, method, budget, output_root):
    model_dir = output_root / "models" / method / f"budget-{budget}"
    model_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model_and_tokenizer(
        suite.sft.cfg.base_model_id, suite.sft.cfg.tokenizer_id, is_inference=False
    )
    tokenized = _tokenize_messages(dataset, tokenizer, suite.sft.cfg.max_seq_length)
    if not suite.sft.cfg.full_ft:
        model = get_peft_model(model, PEFT_CONFIG)
    args = TrainingArguments(
        output_dir=str(model_dir / "trainer"),
        num_train_epochs=1,
        per_device_train_batch_size=suite.sft.params.batch_size,
        learning_rate=suite.sft.cfg.learning_rate,
        lr_scheduler_type=suite.sft.cfg.lr_scheduler_type,
        warmup_steps=0,
        weight_decay=0.01,
        bf16=torch.cuda.is_available(),
        report_to="none",
        seed=suite.seed,
        data_seed=suite.seed,
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=args, train_dataset=tokenized, data_collator=_Collator(tokenizer))
    trainer.train()
    if not suite.sft.cfg.full_ft:
        model = trainer.model.merge_and_unload()
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    return model, tokenizer, model_dir


def run(suite: SuiteConfig, sota: SOTAConfig):
    budgets = suite.prune.params.dataset_sizes
    raw = load_dataset(sota.dataset_id, split=sota.split)
    raw = _messages_dataset(raw, sota.dataset_id, AutoTokenizer.from_pretrained(suite.sft.cfg.tokenizer_id))
    unique_instructions = len({
        next(message["content"] for message in row["messages"] if message["role"] == "user")
        for row in raw
    })
    oversized = [budget for budget in budgets if budget > unique_instructions]
    if oversized:
        raise ValueError(
            f"GRAPE budgets {oversized} exceed the {unique_instructions} unique instructions. "
            "GRAPE selects at most one response per instruction."
        )

    output_root = sota.output_dir or (suite.dirs.sota_results / "grape_s2l")
    output_root.mkdir(parents=True, exist_ok=True)

    target_tokenizer = AutoTokenizer.from_pretrained(suite.sft.cfg.tokenizer_id)
    candidates = _generate_candidates(
        raw, sota.grape.teacher_ids, target_tokenizer, sota.grape.max_tokens, sota.grape.temperature
    )
    target_model = AutoModelForCausalLM.from_pretrained(
        suite.sft.cfg.base_model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    grape_rows = {
        budget: select_grape_responses(
            target_model, target_tokenizer, candidates, budget, sota.grape.batch_size
        )
        for budget in budgets
    }
    del target_model
    torch.cuda.empty_cache()

    proxy_root = output_root / "proxy"
    _train_proxy(raw, suite, sota, proxy_root)
    checkpoint_paths = [proxy_root / f"step-{step}" for step in sota.s2l.trajectory_checkpoints]
    proxy_model, proxy_tokenizer = load_model_and_tokenizer(
        sota.s2l.proxy_model_id, sota.s2l.proxy_tokenizer_id, is_inference=False
    )
    if not suite.sft.cfg.full_ft:
        proxy_model = get_peft_model(proxy_model, PEFT_CONFIG)
    trajectories = collect_checkpoint_trajectories(
        proxy_model,
        proxy_tokenizer,
        raw,
        checkpoint_paths,
        suite.sft.cfg.max_seq_length,
        suite.sft.params.batch_size,
    )
    torch.save(trajectories, output_root / "s2l_trajectories.pt")
    s2l_rows = {
        budget: raw.select(
            select_s2l(
                trajectories,
                budget,
                sota.s2l.n_clusters,
                suite.seed,
                sources=raw["source"] if "source" in raw.column_names else None,
            )
        )
        for budget in budgets
    }

    results = []
    for budget in budgets:
        for method, rows in (("grape", grape_rows[budget]), ("s2l", s2l_rows[budget])):
            selected = Dataset.from_list(rows) if method == "grape" else rows
            dataset_path = output_root / "datasets" / method / f"budget-{budget}"
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            selected.save_to_disk(dataset_path)
            model, tokenizer, model_dir = _train_and_evaluate(
                selected, suite, method, budget, output_root
            )
            eval_result, _ = eval_and_save(
                model=model,
                tokenizer=tokenizer,
                suite=suite,
                model_type=ModelType.SOTA,
                dataset_size=budget,
                topk=None,
                gen_cache_dir=model_dir / "generation_cache",
                save_results_and_config=False,
                sota_model_path=str(model_dir),
                sota_eval_output_dir=model_dir / "alpaca_eval",
            )
            results.append({"method": method, "budget": budget, "results": eval_result})

    with open(output_root / "comparison.json", "w") as handle:
        json.dump(results, handle, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser(description="Run direct GRAPE/S2L comparisons")
    parser.add_argument("-c", "--config", required=True, help="Methodology SuiteConfig YAML")
    parser.add_argument("-s", "--sota-config", required=True, help="Separate SOTA YAML")
    args = parser.parse_args()
    run(SuiteConfig.from_yaml(args.config), SOTAConfig.from_yaml(args.sota_config))


if __name__ == "__main__":
    main()
