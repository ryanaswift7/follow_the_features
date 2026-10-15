import argparse
import gc
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from scipy.stats import spearmanr
from sae_lens import SAE
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer

from src.prune import get_formatting_function, get_topk_features
from src.utils.ensure_pad_token import ensure_pad_token_hooked_transformer


DATASETS = [
    {
        "label": "UltraChat 200K",
        "slug": "ultrachat_200k",
        "dataset_id": "HuggingFaceH4/ultrachat_200k",
        "split": "train_sft",
        "num_samples": 207865,
        "chat_tokenizer_id": "meta-llama/Llama-3.1-8B-Instruct",
    },
    {
        "label": "Dolci 100K",
        "slug": "dolci_100k",
        "dataset_id": "ryanaswift7/dolci-instruct-sft-no-tools-100k",
        "split": "train",
        "num_samples": 100000,
        "chat_tokenizer_id": "allenai/Olmo-3-7B-Instruct",
    },
    {
        "label": "Alpaca",
        "slug": "alpaca",
        "dataset_id": "yahma/alpaca-cleaned",
        "split": "train",
        "num_samples": 51760,
        "chat_tokenizer_id": "HuggingFaceTB/SmolLM3-3B",
    },
]

PRUNE_MODEL_ID = "google/gemma-3-12b-it"
SAE_RELEASE = "gemma-scope-2-12b-it-res"
SAE_ID = "layer_24_width_65k_l0_medium"
PRUNE_BATCH_SIZE = 32
MAX_SEQ_LENGTH = 256
TOP_K_VALUES = [5, 10, 50, 200, 1000, "all"]


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


def _prepare_tokenized_dataset(dataset_cfg, prune_tokenizer):
    dataset = load_dataset(dataset_cfg["dataset_id"], split=dataset_cfg["split"])
    num_samples = dataset_cfg["num_samples"]
    if num_samples is not None and num_samples < len(dataset):
        dataset = dataset.select(range(num_samples))

    chat_tokenizer = AutoTokenizer.from_pretrained(dataset_cfg["chat_tokenizer_id"])
    formatter_func = get_formatting_function(dataset_cfg["dataset_id"])

    str_dataset = dataset.map(
        lambda examples: formatter_func(
            chat_tokenizer, examples, is_inference=False
        ),
        batched=True,
        batch_size=1000,
        num_proc=12,
    )
    chat_str_dataset = str_dataset.map(
        lambda batch: {
            "text": [
                chat_tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                for messages in batch["messages"]
            ]
        },
        batched=True,
        batch_size=1000,
        num_proc=12,
    )
    tokenized_dataset = chat_str_dataset.map(
        lambda examples: prune_tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_tensors=None,
        ),
        batched=True,
        batch_size=1000,
        num_proc=12,
        remove_columns=str_dataset.column_names,
    )
    tokenized_dataset.set_format(type="torch", columns=["input_ids"])
    return tokenized_dataset


def generate_feature_rankings(output_dir: Path, device: str) -> list[Path]:
    """Generate and save full feature rankings without selecting dataset samples."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ranking_paths = []

    model = HookedTransformer.from_pretrained_no_processing(
        PRUNE_MODEL_ID,
        device=device,
        dtype=torch.bfloat16,
    )
    model, prune_tokenizer = ensure_pad_token_hooked_transformer(
        model=model,
        is_inference=True,
    )
    sae, _, _ = SAE.from_pretrained_with_cfg_and_sparsity(
        release=SAE_RELEASE,
        sae_id=SAE_ID,
        device=device,
    )
    hook_name = sae.cfg.metadata.hook_name
    layer_idx = int(hook_name.split(".")[1])

    for dataset_cfg in DATASETS:
        label = dataset_cfg["label"]
        slug = dataset_cfg["slug"]
        print(f"\n[RANKING] Generating feature ranking for {label}")
        tokenized_dataset = _prepare_tokenized_dataset(dataset_cfg, prune_tokenizer)
        dataloader = torch.utils.data.DataLoader(
            tokenized_dataset,
            batch_size=PRUNE_BATCH_SIZE,
            shuffle=False,
        )

        dataset_output_dir = output_dir / "feature_rankings" / slug
        full_report_path = dataset_output_dir / "full_feature_report.csv"
        topk_report_path = dataset_output_dir / "topk_feature_report.json"

        # This runs the regular feature-importance ranking process. It does
        # not call get_top_sample_indices or create a pruned dataset.
        get_topk_features(
            dataloader=dataloader,
            model=model,
            sae=sae,
            hook_name=hook_name,
            layer_idx=layer_idx,
            k=max(k for k in TOP_K_VALUES if isinstance(k, int)),
            device=device,
            divide_by_entropy=True,
            diversity_param=None,
            diversity_seed=0,
            diversity_tag="divoff",
            full_report_path_csv=full_report_path,
            topk_report_path_json=topk_report_path,
            llm_id=PRUNE_MODEL_ID,
            sae_release=SAE_RELEASE,
            sae_id=SAE_ID,
        )
        ranking_paths.append(full_report_path)

        with open(dataset_output_dir / "dataset_config.json", "w") as file:
            json.dump(dataset_cfg, file, indent=2)

        del tokenized_dataset, dataloader
        gc.collect()
        torch.cuda.empty_cache()

    del model, prune_tokenizer, sae
    gc.collect()
    torch.cuda.empty_cache()
    return ranking_paths


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate feature rankings and compare them across datasets."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("feature_ranking_correlation"),
        help="Directory for rankings and correlation results.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for ranking, defaulting to CUDA when available.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ranking_paths = generate_feature_rankings(args.output_dir, args.device)
    labels = [dataset_cfg["label"] for dataset_cfg in DATASETS]

    results_df = compare_top_k_spearman_large_scale(
        csv_paths=ranking_paths,
        top_k_values=TOP_K_VALUES,
        labels=labels,
        feature_col="feature_idx",
    )
    if results_df.empty:
        raise RuntimeError("No feature rankings were available for correlation analysis.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "feature_ranking_correlation_results.csv"
    markdown_path = args.output_dir / "feature_ranking_correlation_results.md"
    results_df.to_csv(csv_path, index=False)
    markdown_path.write_text(results_df.to_markdown(index=False) + "\n")

    print("\nResults:")
    print(results_df.to_string(index=False))
    print(f"\nSaved correlation CSV to {csv_path}")
    print(f"Saved correlation Markdown to {markdown_path}")


if __name__ == "__main__":
    main()
