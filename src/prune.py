import heapq
import json
import os
from pathlib import Path
from typing import Literal, Optional, cast
import csv
import math
import time

import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch
from datasets import load_dataset
from sae_lens import SAE
from scipy.stats import differential_entropy
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer
import numpy as np

from src.schemas.suite import SuiteConfig
from src.utils.constants import DATASET_CONFIG_FILENAME, PRUNE_ID_KEY
from src.utils.feature_reports import write_full_feature_report_csv, write_topk_feature_report_json


def estimate_vasicek_entropy(vals: np.ndarray):
    """
    Estimates differential entropy using the Vasicek n-spacing method.
    Expects a 1D numpy array of already-filtered positive activations.
    """
    if len(vals) < 20:
        return 1e-5
        
    # Uniform noise is computationally cheaper than a normal distribution
    jitter = np.random.uniform(-1e-8, 1e-8, size=vals.shape)
    vals_jittered = vals + jitter
    
    return differential_entropy(vals_jittered)

def compute_single_importance_task(args):
    """
    Task worker for ProcessPoolExecutor.
    """
    i, mwf, act_slice_np, decoder_norm = args
    h_clipped = 1.0
    
    if act_slice_np is not None:
        h = float(estimate_vasicek_entropy(act_slice_np))
        h_clipped = max(h, 0.01)

    return i, (mwf * decoder_norm) / h_clipped


def get_topk_features(
    dataloader,
    model,
    sae,
    hook_name,
    layer_idx,
    k,
    divide_by_entropy: bool,
    full_report_path_csv: Path,
    topk_report_path_json: Path,
    llm_id: str,
    sae_release: str,
    sae_id: str,
    max_entropy_samples: int = 1024,
    device="cuda",
    refine_top_n: int = 10_000,
):

    # ==========================================
    # PATH SETUP
    # ==========================================
    csv_report_path = full_report_path_csv.with_suffix(".csv")
    json_topk_path = topk_report_path_json.with_suffix(".json")

    def log(msg):
        print(f"[get_topk_features] {msg}", flush=True)

    t0_total = time.time()

    d_sae = sae.cfg.d_sae
    log(f"Starting feature extraction")
    log(f"d_sae={d_sae:,} | refine_top_n={refine_top_n:,} | max_entropy_samples={max_entropy_samples}")

    # ==========================================
    # PASS 1: STATS ONLY (GPU FAST)
    # ==========================================
    t0 = time.time()
    log("Phase 1: Forward pass (stats only)")

    feature_sum = torch.zeros(d_sae, device=device)
    feature_sq_sum = torch.zeros(d_sae, device=device)
    feature_counts = torch.zeros(d_sae, device=device)

    total_tokens = 0
    total_nonzero = 0

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Forward Pass 1")):
        tokens = batch["input_ids"].to(device)

        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=[hook_name],
                stop_at_layer=layer_idx + 1,
            )

            acts = cache[hook_name]
            flat = acts.view(-1, acts.shape[-1])
            total_tokens += flat.shape[0]

            scaled = flat * torch.rsqrt(
                flat.pow(2).mean(dim=-1, keepdim=True) + 1e-8
            )
            feats = sae.encode(scaled)

            mask = feats > 0
            nz = feats * mask

            nonzero_count = mask.sum().item()
            total_nonzero += nonzero_count

            feature_sum += nz.sum(dim=0)
            feature_sq_sum += (nz ** 2).sum(dim=0)
            feature_counts += mask.sum(dim=0)

        if (batch_idx + 1) % 10 == 0:
            sparsity = total_nonzero / max(total_tokens * d_sae, 1)
            log(f"Batch {batch_idx+1}: tokens={total_tokens:,}, nnz={total_nonzero:,}, sparsity≈{sparsity:.2e}")

    log(f"Phase 1 complete in {time.time() - t0:.2f}s")

    # ==========================================
    # COARSE IMPORTANCE (GPU)
    # ==========================================
    t0 = time.time()
    log("Computing coarse importance")

    safe_counts = feature_counts.clamp(min=1)
    mean = feature_sum / safe_counts
    var = (feature_sq_sum / safe_counts) - mean**2
    var = var.clamp(min=1e-8)

    decoder_norms = torch.norm(sae.W_dec, p=2, dim=-1)

    if divide_by_entropy:
        entropy = 0.5 * torch.log(2 * math.pi * math.e * var)
        coarse_importance = (mean * decoder_norms) / entropy.clamp(min=1e-8)
    else:
        coarse_importance = mean * decoder_norms

    log(f"Coarse importance computed in {time.time() - t0:.2f}s")

    # ==========================================
    # SELECT TOP-N
    # ==========================================
    t0 = time.time()
    log(f"Selecting top {refine_top_n:,}")

    _, top_idx = torch.topk(coarse_importance, refine_top_n)
    top_idx_np = top_idx.cpu().numpy()

    idx_map = {f: i for i, f in enumerate(top_idx_np)}

    log(f"Top-N selection done in {time.time() - t0:.2f}s")

    # ==========================================
    # PASS 2: SAMPLE ONLY TOP FEATURES (SAFE)
    # ==========================================
    t0 = time.time()
    log("Phase 2: Sampling top features (second pass)")

    sample_buffer = np.zeros((refine_top_n, max_entropy_samples), dtype=np.float32)
    sample_counts = np.zeros(refine_top_n, dtype=np.int32)

    top_idx_set = set(top_idx_np.tolist())

    for batch in tqdm(dataloader, desc="Forward Pass 2 (Sampling)"):
        tokens = batch["input_ids"].to(device)

        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=[hook_name],
                stop_at_layer=layer_idx + 1,
            )

            acts = cache[hook_name]
            flat = acts.view(-1, acts.shape[-1])

            scaled = flat * torch.rsqrt(
                flat.pow(2).mean(dim=-1, keepdim=True) + 1e-8
            )
            feats = sae.encode(scaled)

            nz_idx = (feats > 0).nonzero(as_tuple=True)
            if len(nz_idx[0]) == 0:
                continue

            vals = feats[nz_idx].cpu().numpy()
            feat_ids = nz_idx[1].cpu().numpy()

            for f, v in zip(feat_ids, vals):
                if f not in top_idx_set:
                    continue

                i = idx_map[f]
                c = sample_counts[i]

                if c < max_entropy_samples:
                    sample_buffer[i, c] = v
                    sample_counts[i] += 1

    log(f"Sampling complete in {time.time() - t0:.2f}s")

    # ==========================================
    # VASICEK REFINEMENT
    # ==========================================
    t0 = time.time()
    log("Computing Vasicek entropy")

    importance = coarse_importance.cpu().numpy()
    decoder_norms_np = decoder_norms.cpu().numpy()

    valid_entropy = 0

    for i in tqdm(range(refine_top_n), desc="Vasicek"):
        count = sample_counts[i]

        if divide_by_entropy and count > 20:
            vals = sample_buffer[i, :count]
            h = estimate_vasicek_entropy(vals)
            h = max(h, 0.01)
            valid_entropy += 1
        else:
            h = 1.0

        f = top_idx_np[i]
        mwf = (feature_sum[f] / max(feature_counts[f], 1)).item()
        importance[f] = (mwf * decoder_norms_np[f]) / h

    log(f"Vasicek done in {time.time() - t0:.2f}s | computed={valid_entropy:,}")

    # ==========================================
    # FINAL SORT
    # ==========================================
    t0 = time.time()
    log("Final full sort")

    sorted_feature_indices = np.argsort(-importance)
    sorted_importance_raw = importance[sorted_feature_indices]

    log(f"Sort done in {time.time() - t0:.2f}s")

    # ==========================================
    # WRITE REPORTS (UNCHANGED PIPELINE)
    # ==========================================
    log(f"Writing CSV → {csv_report_path}")

    full_report_data = write_full_feature_report_csv(
        feat_indices=sorted_feature_indices,
        feat_values=sorted_importance_raw,
        total_features=d_sae,
        llm_id=llm_id,
        sae_release=sae_release,
        sae_id=sae_id,
        hook_name=hook_name,
        report_path=csv_report_path,
    )

    log(f"Writing JSON → {json_topk_path}")

    write_topk_feature_report_json(
        full_report_data=full_report_data,
        k=k,
        topk_report_path=json_topk_path,
    )

    log(f"TOTAL TIME: {time.time() - t0_total:.2f}s")

    return (
        sorted_feature_indices[:k].tolist(),
        sorted_importance_raw[:k].tolist(),
    )

def get_top_sample_indices(
    dataloader,
    model,
    sae,
    hook_name,
    layer_idx,
    feature_indices,
    n_samples,
    sequence_aggregation: Literal["sum", "mean", "max"],
    feature_aggregation: Literal["sum", "mean"],
    device="cuda",
):
    """
    Ranks data samples using a Min-Heap for efficient streaming Top-N selection.

    Returns:
        list[int]: Sorted list of dataset indices (highest scores first).
    """

    # Ensure indices are on the correct device
    if not isinstance(feature_indices, torch.Tensor):
        feature_indices = torch.tensor(feature_indices, device=device)
    else:
        feature_indices = feature_indices.to(device)

    # The Heap: Stores tuples of (score, dataset_index)
    # Python's heapq is a min-heap. We will keep the 'top n' largest scores.
    top_heap = []

    print(
        f"Scanning dataset (Heap Optimized). Aggregation: Seq='{sequence_aggregation}', Feat='{feature_aggregation}'..."
    )

    dataset_start_index = 0

    # Determine total batches if possible for the progress bar
    total_batches = len(dataloader) if hasattr(dataloader, "__len__") else None

    # Wrap dataloader in tqdm
    for i, batch in enumerate(
        tqdm(dataloader, total=total_batches, desc="Scanning Dataset", unit="batch")
    ):
        batch_len = len(batch["input_ids"])
        batch_indices = range(dataset_start_index, dataset_start_index + batch_len)

        tokens = batch["input_ids"].to(device)

        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens, names_filter=[hook_name], stop_at_layer=layer_idx + 1
            )
            acts = cache[hook_name]

            # 1. Gemma Scope Unit Norm Scaling
            rms = torch.sqrt(torch.mean(acts**2, dim=-1, keepdim=True) + 1e-6)
            scaled_acts = acts / rms

            # 2. Encode
            flat_acts = scaled_acts.view(-1, acts.shape[-1])
            feature_acts = sae.encode(flat_acts)
            feature_acts = feature_acts.view(acts.shape[0], acts.shape[1], -1)

            # 3. Filter
            relevant_acts = feature_acts[:, :, feature_indices]

            # 4. Sequence Aggregation
            if sequence_aggregation == "sum":
                seq_scores = relevant_acts.sum(dim=1)
            elif sequence_aggregation == "mean":
                seq_scores = relevant_acts.mean(dim=1)
            elif sequence_aggregation == "max":
                seq_scores = relevant_acts.max(dim=1).values
            else:
                raise ValueError("sequence_aggregation must be 'sum', 'mean', or 'max'")

            # 5. Feature Aggregation
            if feature_aggregation == "sum":
                final_scores = seq_scores.sum(dim=1)
            elif feature_aggregation == "mean":
                final_scores = seq_scores.mean(dim=1)
            else:
                raise ValueError("feature_aggregation must be 'sum' or 'mean'")

            # Move to CPU
            final_scores = final_scores.cpu().tolist()

            # --- HEAP OPTIMIZATION ---
            for score, idx in zip(final_scores, batch_indices):
                if len(top_heap) < n_samples:
                    # If buffer isn't full, just push
                    heapq.heappush(top_heap, (score, idx))
                else:
                    # Buffer is full.
                    # heappushpop pushes the new item, then pops the smallest item.
                    # This maintains the "Top N" efficiently.
                    heapq.heappushpop(top_heap, (score, idx))

        # Cleanup
        del (
            tokens,
            acts,
            scaled_acts,
            feature_acts,
            relevant_acts,
            seq_scores,
            final_scores,
        )
        torch.cuda.empty_cache()

        dataset_start_index += batch_len

    # --- Final Sort ---
    # The heap is not sorted; it just guarantees heap[0] is the minimum.
    # We sort descending by score to return the "Best" samples first.
    top_heap.sort(key=lambda x: x[0], reverse=True)

    return [idx for score, idx in top_heap]


def format_alpaca(tokenizer, examples, is_inference=False):
    # Safely get columns, defaulting to empty strings if missing
    length = len(examples.get("output", examples.get("instruction", [])))
    instructions = examples.get("instruction", [""] * length)
    inputs = examples.get("input", [""] * length)
    outputs = examples.get("output", [""] * length)
    
    batch_conversations = []
    
    for inst, inp, out in zip(instructions, inputs, outputs):
        inst_stripped = inst.strip()
        inp_stripped = inp.strip()
        
        # Prepend instruction before the user input
        if inp_stripped:
            user_content = f"{inst_stripped}\n\n{inp_stripped}"
        else:
            user_content = inst_stripped
            
        conversation = [{"role": "user", "content": user_content}]
        
        # Append assistant response if we are training
        if not is_inference and out:
            conversation.append({"role": "assistant", "content": out.strip()})
            
        batch_conversations.append(conversation)
        
    # Apply template to the ENTIRE batch at once
    formatted_texts = tokenizer.apply_chat_template(
        batch_conversations, 
        tokenize=False, 
        add_generation_prompt=is_inference
    )
        
    return {"text": formatted_texts}


def format_standard_messages(tokenizer, examples, is_inference=False):
    has_prompt = "prompt" in examples
    has_instruction = "instruction" in examples
    
    batch_conversations = []
    
    for i, messages in enumerate(examples["messages"]):
        # Create a shallow copy of the message list
        prompt_messages = list(messages)
        
        # Gather anything that needs to be prepended
        prepends = []
        if has_instruction and examples["instruction"][i]:
            prepends.append(examples["instruction"][i].strip())
        if has_prompt and examples["prompt"][i]:
            prepends.append(examples["prompt"][i].strip())
            
        # Prepend to the first user message
        if prepends and prompt_messages and prompt_messages[0]["role"] == "user":
            combined_prepend = "\n\n".join(prepends)
            current_content = prompt_messages[0]["content"]
            
            # Safeguard: Don't prepend if the text is already identical
            if combined_prepend not in current_content:
                # Create a new dict for the modified message to avoid mutating original
                first_msg = prompt_messages[0].copy()
                first_msg["content"] = f"{combined_prepend}\n\n{current_content}"
                prompt_messages[0] = first_msg
        
        # Strip final assistant turn if doing inference
        if is_inference and prompt_messages and prompt_messages[-1]["role"] == "assistant":
            prompt_messages = prompt_messages[:-1]
            
        batch_conversations.append(prompt_messages)
            
    # Apply template to the ENTIRE batch at once
    formatted_texts = tokenizer.apply_chat_template(
        batch_conversations, 
        tokenize=False, 
        add_generation_prompt=is_inference
    )
        
    return {"text": formatted_texts}


def get_formatting_function(dataset_id: str):
    """
    Returns a formatting function tailored to the specific dataset structure.
    """
    dataset_id_lower = dataset_id.lower()
    
    # Map keywords from the dataset repo name to the correct function
    if "alpaca" in dataset_id_lower:
        return format_alpaca
    elif "dolci" in dataset_id_lower or "ultrachat" in dataset_id_lower:
        return format_standard_messages
    else:
        raise ValueError(
            f"Unsupported dataset ID '{dataset_id}'. "
            "Please add a custom parser for this dataset structure."
        )


def run_prune(suite: SuiteConfig, dataset_size: int, topk: int):
    print(f"[PRUNE] Starting pruning for dataset_size={dataset_size}, topk={topk}...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Unpack config
    model_id = suite.prune.cfg.prune_model_id
    sae_release = suite.prune.cfg.sae_release
    sae_id = suite.prune.cfg.sae_id
    dataset_id = suite.prune.cfg.dataset_id
    batch_size = suite.prune.params.batch_size
    num_samples = suite.prune.cfg.num_samples
    seq_agg_method = suite.prune.cfg.seq_agg_method
    feat_agg_method = suite.prune.cfg.feat_agg_method

    print(f"--- Prune: dataset_size={dataset_size}, topk={topk} ---")
    tokenizer = AutoTokenizer.from_pretrained(suite.sft.cfg.tokenizer_id, fix_mistral_regex=True)
    model = HookedTransformer.from_pretrained_no_processing(
        model_id, device=device, dtype=torch.bfloat16
    )
    sae, _, _ = SAE.from_pretrained_with_cfg_and_sparsity(
        release=sae_release, sae_id=sae_id, device=device
    )
    if "ultrachat" in dataset_id:
        # handle different split name in ultrachat 200k
        dataset = load_dataset(dataset_id, split="train_sft")
    else:
        dataset = load_dataset(dataset_id, split="train")

    if num_samples is not None and num_samples < len(dataset):
        dataset = dataset.select(range(num_samples))

    formatter_func = get_formatting_function(dataset_id)

    str_dataset = dataset.map(
        lambda examples: formatter_func(tokenizer, examples, is_inference=False),
        batched=True,
        batch_size=1000,  # batch_size,
        num_proc=12,
    )

    tokenized_dataset = str_dataset.map(
        lambda examples: tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=suite.sft.cfg.max_seq_length,
            return_tensors=None  # .map will handle conversion
        ),
        batched=True,
        batch_size=1000,
        num_proc=12,
        remove_columns=str_dataset.column_names 
    )
    tokenized_dataset.set_format(type="torch", columns=["input_ids"])


    feature_selection_dataloader = torch.utils.data.DataLoader(
        tokenized_dataset, batch_size=batch_size, shuffle=False,
    )
    hook_name = sae.cfg.metadata.hook_name
    layer_idx = int(hook_name.split(".")[1])

    top_indices, top_values = get_topk_features(
        dataloader=feature_selection_dataloader,
        model=model,
        sae=sae,
        hook_name=hook_name,
        layer_idx=layer_idx,
        k=topk,
        device=device,
        divide_by_entropy=suite.prune.cfg.divide_by_entropy,
        full_report_path_csv=suite.dirs.datasets / "full_feature_report.csv",
        topk_report_path_json=suite.dirs.get_dataset_path(
            dataset_size=dataset_size, topk=topk) / "topk_feature_report.json",
        llm_id=suite.prune.cfg.prune_model_id,
        sae_release=suite.prune.cfg.sae_release,
        sae_id=suite.prune.cfg.sae_id,
    )


    sample_selection_dataloader = torch.utils.data.DataLoader(
        tokenized_dataset, batch_size=batch_size, shuffle=False,
    )

    kept_indices = get_top_sample_indices(
        sample_selection_dataloader,
        model,
        sae,
        hook_name,
        layer_idx,
        top_indices,
        n_samples=dataset_size,
        sequence_aggregation=seq_agg_method,
        feature_aggregation=feat_agg_method,
        device=device,
    )

    assert kept_indices, "No samples were kept after pruning!"
    pruned_dataset = str_dataset.select(kept_indices)
    pruned_dataset.save_to_disk(
        suite.dirs.get_dataset_path(dataset_size=dataset_size, topk=topk)
    )

    dataset_cfg_path = os.path.join(
        suite.dirs.get_dataset_path(dataset_size=dataset_size, topk=topk),
        DATASET_CONFIG_FILENAME,
    )
    model_cfg = {
        PRUNE_ID_KEY: suite.ids.prune,
    }
    with open(dataset_cfg_path, "w") as f:
        json.dump(model_cfg, f, indent=4)

    print(f"[PRUNE] Dataset generated for dataset_size={dataset_size}, topk={topk}.")
