import heapq
import json
import os
from pathlib import Path
from typing import Literal, Optional, cast
import csv

import concurrent.futures
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


def estimate_vasicek_entropy(activations, window=None):
    """
    Estimates differential entropy using the Vasicek n-spacing method.
    Includes jitter to prevent divide-by-zero errors from duplicate values.
    """
    # Filter for positive activations and convert to numpy
    vals = activations[activations > 0].numpy()
    
    if len(vals) < 20:  # Minimum samples for a stable estimate
        return 1e-5
        
    # Add microscopic noise (jitter) to break exact ties
    noise_scale = 1e-8
    jitter = np.random.normal(0, noise_scale, size=vals.shape)
    vals_jittered = vals + jitter
    
    return differential_entropy(vals_jittered)


def estimate_vasicek_entropy_vectorized(
    activations: torch.Tensor, m: Optional[int] = None
) -> float:
    """
    Estimates differential entropy using the Vasicek n-spacing method.
    Robust implementation in pure PyTorch (avoids CPU/Numpy overhead).

    Formula: H(X) = 1/n * sum( ln( n/2m * (x_{i+m} - x_{i-m}) ) )
    """
    # 1. Robust Filtering & Casting
    # Ensure float32 for log stability (activations often come in as bfloat16)
    if activations.dtype == torch.bfloat16:
        activations = activations.float()

    # Filter zeros to analyze the active distribution only
    vals = activations[activations > 0]

    n = vals.numel()
    # Require a minimum number of samples for a statistically valid density estimate
    if n < 20:
        return 1e-5

    # 2. Sort (Required for spacing methods)
    # PyTorch sort is highly optimized and often faster than converting to numpy
    vals, _ = torch.sort(vals)

    # 3. Determine Window Size (m)
    # Heuristic: sqrt(n) is the standard optimal choice for Vasicek
    if m is None:
        m = int(n**0.5)

    # Clamp m to ensure valid spacing (must be < n/2 for the 2m logic)
    m = min(max(1, m), n // 2)

    # 4. Vectorized Gap Calculation using Padding
    # We want x_{i+m} - x_{i-m} for all i.
    # To vectorize, we pad the sorted array at both ends:
    # Left pad: repeat x_0 (m times)
    # Right pad: repeat x_{n-1} (m times)
    pad_left = vals[0].expand(m)
    pad_right = vals[-1].expand(m)

    padded = torch.cat([pad_left, vals, pad_right])

    # Calculate gaps: padded[i + 2m] - padded[i]
    # This aligns the window perfectly for the entire tensor in one operation
    gaps = padded[2 * m :] - padded[: -2 * m]

    # 5. Compute Entropy
    # Add epsilon to prevent log(0) if feature values are identical (e.g., clamped/discrete)
    eps = 1e-6
    gaps = torch.clamp(gaps, min=eps)

    # H = mean( log( (n / 2m) * gaps ) )
    #   = log(n/2m) + mean(log(gaps))
    const_factor = n / (2 * m)
    entropy = torch.log(gaps * const_factor).mean()

    return entropy.item()

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
    max_entropy_samples: int = 50_000,
    device="cuda",
):
    # Enforce extensions for clarity
    csv_report_path = full_report_path_csv.with_suffix('.csv')
    json_topk_path = topk_report_path_json.with_suffix('.json')

    # ==========================================
    # 1. Check if the ranking already exists
    # ==========================================
    if csv_report_path.exists():
        print(f"[CACHE] Complete feature report found at {csv_report_path}. Extracting top {k} features...")
        
        top_feature_indices = []
        top_importance_raw = []
        full_report_dict_for_topk = {"cfg": {}, "features": {}}
        
        # Read the CSV to reconstruct the dict for the top-k subset
        with open(csv_report_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            count = 0
            headers_found = False
            
            for row in reader:
                if not row:
                    continue
                    
                # Parse config comments
                if row[0].startswith("#"):
                    line = row[0][1:].strip()
                    if ":" in line:
                        key, val = line.split(":", 1)
                        full_report_dict_for_topk["cfg"][key.strip()] = val.strip()
                    continue
                
                # Skip the header row
                if not headers_found:
                    headers_found = True
                    continue
                
                # Parse data rows (only need up to k)
                if count < k:
                    idx_str = row[0]
                    raw_val = row[1]
                    
                    top_feature_indices.append(int(idx_str))
                    top_importance_raw.append(float(raw_val))
                    
                    # Reconstruct the dict structure for the Top-K JSON writer
                    full_report_dict_for_topk["features"][idx_str] = {
                        "value": {
                            "raw": raw_val,
                            "norm_sum": row[2],
                            "norm_max": row[3],
                            "zscore": row[4],
                        }
                    }
                    count += 1
                else:
                    break # We have our top k, stop reading the 65k CSV

        # Ensure the top-k report also exists if loaded from cache
        if not json_topk_path.exists():
            write_topk_feature_report_json(full_report_dict_for_topk, k, json_topk_path)

        return top_feature_indices, top_importance_raw

    # ==========================================
    # 2. Compute rankings (if file not found)
    # ==========================================
    d_sae = sae.cfg.d_sae
    feature_sum = torch.zeros(d_sae, device="cpu", dtype=torch.float32)

    if divide_by_entropy:
        sample_buffer = torch.zeros((d_sae, max_entropy_samples), dtype=torch.bfloat16, device="cpu")
        sample_counts = torch.zeros(d_sae, dtype=torch.long, device="cpu")

    total_samples_processed = 0
    print(f"Computing feature importance for all {d_sae} features...")

    total_batches = len(dataloader) if hasattr(dataloader, "__len__") else None

    for batch in tqdm(dataloader, total=total_batches, desc="Processing Batches", unit="batch"):
        tokens = model.to_tokens(batch["text"], truncate=True).to(device)

        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens, names_filter=[hook_name], stop_at_layer=layer_idx + 1
            )
            acts = cache[hook_name]
            flat_acts = acts.reshape(-1, acts.shape[-1])

            rms = torch.sqrt(torch.mean(flat_acts**2, dim=-1, keepdim=True) + 1e-8)
            scaled_acts = flat_acts / rms

            feature_acts = sae.encode(scaled_acts)

            feature_sum += feature_acts.sum(dim=0).to(dtype=torch.float32, device="cpu")
            total_samples_processed += flat_acts.shape[0]

            if divide_by_entropy:
                mask = feature_acts > 0
                nonzero_indices = mask.nonzero(as_tuple=True)
                token_idxs, feat_idxs = nonzero_indices
                nonzero_vals = feature_acts[token_idxs, feat_idxs]

                feat_idxs_cpu = feat_idxs.cpu()
                nonzero_vals_cpu = nonzero_vals.to(dtype=torch.bfloat16, device="cpu")

                if feat_idxs_cpu.numel() > 0:
                    sort_idx = torch.argsort(feat_idxs_cpu)
                    sorted_feats = feat_idxs_cpu[sort_idx]
                    sorted_vals = nonzero_vals_cpu[sort_idx]

                    unique_feats, counts = torch.unique_consecutive(sorted_feats, return_counts=True)
                    split_indices = torch.cumsum(counts, dim=0)
                    split_indices = torch.cat([torch.tensor([0], device="cpu"), split_indices])

                    for idx_in_unique, f_idx_tensor in enumerate(unique_feats):
                        f_idx = f_idx_tensor.item()
                        current_count = sample_counts[f_idx].item()

                        if current_count < max_entropy_samples:
                            start = split_indices[idx_in_unique].item()
                            end = split_indices[idx_in_unique + 1].item()
                            
                            space_left = max_entropy_samples - current_count
                            to_add = sorted_vals[start:end][:space_left]
                            num_to_add = to_add.size(0)

                            sample_buffer[f_idx, current_count : current_count + num_to_add] = to_add
                            sample_counts[f_idx] += num_to_add
        # break # Assuming this break is intentional for testing

    print("\nCalculating Entropy and Importance Scores...")
    decoder_norms = torch.norm(sae.W_dec, p=2, dim=-1).cpu()
    final_importance = torch.zeros(d_sae)

    def compute_single_importance(i):
        mwf = feature_sum[i] / total_samples_processed
        h_clipped = 1.0

        if divide_by_entropy:
            count = sample_counts[i].item()
            if count > 0:
                acts_tensor = sample_buffer[i, :count].to(torch.float32)
                h = cast(float, estimate_vasicek_entropy(acts_tensor))
            else:
                h = 1.0
            h_clipped = max(h, 0.01)

        return i, (mwf * decoder_norms[i]) / h_clipped

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(compute_single_importance, i): i for i in range(d_sae)}
        for future in tqdm(concurrent.futures.as_completed(futures), total=d_sae, desc="Calculating Entropy"):
            i, importance_val = future.result()
            final_importance[i] = importance_val
            
    # Sort ALL features descending
    sorted_importance_raw, sorted_feature_indices = torch.sort(final_importance, descending=True)

    # ==========================================
    # 3. Write out the Reports
    # ==========================================
    
    # Report 1: The Complete Feature Ranking (as CSV)
    full_report_data = write_full_feature_report_csv(
        feat_indices=sorted_feature_indices,
        feat_values=sorted_importance_raw,
        total_features=d_sae,
        llm_id=llm_id,
        sae_release=sae_release, 
        sae_id=sae_id,
        hook_name=hook_name,
        report_path=csv_report_path
    )

    # Report 2: Extract Top-K into JSON
    write_topk_feature_report_json(
        full_report_data=full_report_data, 
        k=k, 
        topk_report_path=json_topk_path
    )

    # ==========================================
    # 4. Return ONLY the requested top-k
    # ==========================================
    return (
        sorted_feature_indices[:k].tolist(),
        sorted_importance_raw[:k].tolist()
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
        batch_len = len(batch["text"])
        batch_indices = range(dataset_start_index, dataset_start_index + batch_len)

        tokens = model.to_tokens(batch["text"], truncate=True).to(device)

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

    dataset = dataset.map(
        lambda examples: formatter_func(tokenizer, examples, is_inference=False),
        batched=True,
        batch_size=1000,  # batch_size,
        num_proc=12,

    )
    feature_selection_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size
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
        dataset, batch_size=batch_size
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
    pruned_dataset = dataset.select(kept_indices)
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
