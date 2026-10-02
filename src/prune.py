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
from src.utils.ensure_pad_token import ensure_pad_token, ensure_pad_token_hooked_transformer


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


def apply_feature_diversity(
    sorted_feature_indices,
    sorted_importance_raw,
    k: int,
    diversity_param: Optional[float],
    seed: int,
):
    if k <= 0:
        return [], []

    k = min(k, len(sorted_feature_indices))
    if k == 0:
        return [], []

    if diversity_param is None or diversity_param <= 0.0:
        selected_feature_indices = sorted_feature_indices[:k].tolist()
        selected_feature_values = sorted_importance_raw[:k].tolist()
        return selected_feature_indices, selected_feature_values

    replace_count = min(int(k * diversity_param), k)
    if replace_count <= 0:
        selected_feature_indices = sorted_feature_indices[:k].tolist()
        selected_feature_values = sorted_importance_raw[:k].tolist()
        return selected_feature_indices, selected_feature_values

    keep_count = k - replace_count
    kept_indices = sorted_feature_indices[:keep_count]
    kept_values = sorted_importance_raw[:keep_count]
    replacement_pool = sorted_feature_indices[k:]
    fallback_tail = sorted_feature_indices[keep_count:k]

    rank_lookup = {int(idx): pos for pos, idx in enumerate(sorted_feature_indices.tolist())}
    combined_indices = kept_indices.tolist()

    if len(replacement_pool) > 0:
        sample_count = min(replace_count, len(replacement_pool))
        rng = np.random.default_rng(seed)
        sampled_positions = rng.choice(len(replacement_pool), size=sample_count, replace=False)
        sampled_indices = replacement_pool[sampled_positions]
        combined_indices.extend(int(idx) for idx in sampled_indices.tolist())
    else:
        sample_count = 0

    if sample_count < replace_count and len(fallback_tail) > 0:
        fallback_needed = min(replace_count - sample_count, len(fallback_tail))
        combined_indices.extend(int(idx) for idx in fallback_tail[:fallback_needed].tolist())

    combined_indices.sort(key=lambda idx: rank_lookup[idx])
    combined_values = [float(sorted_importance_raw[rank_lookup[idx]]) for idx in combined_indices]
    return combined_indices, combined_values


def get_topk_features(
    dataloader,
    model,
    sae,
    hook_name,
    layer_idx,
    k,
    divide_by_entropy: bool,
    diversity_param: Optional[float],
    diversity_seed: int,
    diversity_tag: str,
    full_report_path_csv: Path,
    topk_report_path_json: Path,
    llm_id: str,
    sae_release: str,
    sae_id: str,
    max_entropy_samples: int = 2048,
    device="cuda",
    refine_top_n: int = 1_000,
    chunk_size: int = 256,  # prevents the 1-million-dim expansion from blowing up VRAM

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
    log("Starting feature extraction")
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

            # 1. Gemma Scope Unit Norm Scaling
            rms = torch.sqrt(torch.mean(acts**2, dim=-1, keepdim=True) + 1e-6)
            scaled_acts = acts / rms
            
            flat = acts.view(-1, acts.shape[-1])
            total_tokens += flat.shape[0]

            for start_idx in range(0, flat.shape[0], chunk_size):
                end_idx = min(start_idx + chunk_size, flat.shape[0])
                acts_chunk = flat[start_idx:end_idx]
                
                # Let the SAE (and SAELens) handle the raw activations natively
                feats = sae.encode(acts_chunk)

                mask = feats > 0
                nz = feats * mask

                total_nonzero += mask.sum().item()

                feature_sum += nz.sum(dim=0)
                feature_sq_sum += (nz ** 2).sum(dim=0)
                feature_counts += mask.sum(dim=0)
                
                # Explicitly free chunk memory
                del feats, mask, nz
        # break

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

    # top_idx stays on the GPU for Phase 2 slicing
    _, top_idx = torch.topk(coarse_importance, refine_top_n)
    
    # We still need the numpy version for Phase 3 and final mapping
    top_idx_np = top_idx.cpu().numpy()

    log(f"Top-N selection done in {time.time() - t0:.2f}s")

    # ==========================================
    # PASS 2: SAMPLE ONLY TOP FEATURES (ZERO SYNC)
    # ==========================================
    t0 = time.time()
    log("Phase 2: Sampling top features (second pass - pure GPU)")

    # Allocate buffers entirely on the GPU. 
    # Flattened 1D buffer allows for direct advanced indexing assignment.
    gpu_buffer = torch.zeros(refine_top_n * max_entropy_samples, device=device, dtype=torch.float32)
    gpu_counts = torch.zeros(refine_top_n, dtype=torch.long, device=device)

    # Pre-compute row offsets to avoid recalculating inside the loop
    row_offsets = (torch.arange(refine_top_n, device=device) * max_entropy_samples).unsqueeze(1)

    for batch in tqdm(dataloader, desc="Forward Pass 2 (Sampling)"):
        tokens = batch["input_ids"].to(device)

        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=[hook_name],
                stop_at_layer=layer_idx + 1,
            )

            acts = cache[hook_name]

            # 1. Gemma Scope Unit Norm Scaling
            rms = torch.sqrt(torch.mean(acts**2, dim=-1, keepdim=True) + 1e-6)
            scaled_acts = acts / rms

            flat = acts.view(-1, scaled_acts.shape[-1])

            # Still omitting the manual scaling for Gemma Scope 2 compatibility
            for start_idx in range(0, flat.shape[0], chunk_size):
                end_idx = min(start_idx + chunk_size, flat.shape[0])
                acts_chunk = flat[start_idx:end_idx]
                chunk_dim = acts_chunk.shape[0]
                
                feats = sae.encode(acts_chunk)

                # Slice to only the top_n features: shape drops to [chunk_size, refine_top_n]
                top_feats = feats[:, top_idx]
                
                # Transpose to [refine_top_n, chunk_size] for row-wise processing
                top_feats_t = top_feats.T
                
                # Count how many actual non-zeros fired per feature in this chunk
                active_counts = (top_feats_t > 0).sum(dim=1)
                
                # Fast path skip if the chunk is completely dead for our top features
                if active_counts.sum() == 0:
                    del feats, top_feats, top_feats_t, active_counts
                    continue

                # Sorting descending packs all positive non-zeros to the front of each row
                sorted_feats, _ = torch.sort(top_feats_t, dim=1, descending=True)

                # Create relative indices [1, chunk_dim] for the sorted items
                j_idx = torch.arange(chunk_dim, device=device).unsqueeze(0)
                
                # Calculate what their target indices will be in the buffer based on current counts
                target_idx = gpu_counts.unsqueeze(1) + j_idx
                
                # Boolean mask: Only take items that are actual non-zeros AND haven't hit the sample cap
                valid_mask = (j_idx < active_counts.unsqueeze(1)) & (target_idx < max_entropy_samples)

                if valid_mask.any():
                    # Map to the global 1D coordinates in the flattened buffer
                    global_target_idx = row_offsets + target_idx
                    
                    # Direct advanced indexing assignment (100% vectorized, pure GPU)
                    gpu_buffer[global_target_idx[valid_mask]] = sorted_feats[valid_mask]
                    
                    # Update counts, hard-clamping at max_entropy_samples
                    gpu_counts = torch.clamp(gpu_counts + active_counts, max=max_entropy_samples)

                # Explicitly free memory
                del feats, top_feats, top_feats_t, sorted_feats, active_counts, j_idx, target_idx, valid_mask
        # break

    # --- THE ONLY SYNC IN PHASE 2 ---
    # Move the fully populated buffer to CPU and reshape to match what Phase 3 expects
    sample_buffer = gpu_buffer.view(refine_top_n, max_entropy_samples).cpu().numpy()
    sample_counts = gpu_counts.cpu().numpy()

    log(f"Sampling complete in {time.time() - t0:.2f}s")

    # ==========================================
    # VASICEK REFINEMENT
    # ==========================================
    t0 = time.time()
    log("Computing Vasicek entropy")

    importance = coarse_importance.cpu().detach().numpy()
    decoder_norms_np = decoder_norms.cpu().detach().numpy()

    valid_entropy = 0

    for i in tqdm(range(refine_top_n), desc="Vasicek"):
        count = sample_counts[i]

        if divide_by_entropy and count > 20:
            vals = sample_buffer[i, :count]
            h = estimate_vasicek_entropy(vals) # Ensure this relies on your optimized version
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
    # SELECT DIVERSE TOP-K
    # ==========================================
    selected_feature_indices, selected_feature_values = apply_feature_diversity(
        sorted_feature_indices=sorted_feature_indices,
        sorted_importance_raw=sorted_importance_raw,
        k=k,
        diversity_param=diversity_param,
        seed=diversity_seed,
    )

    # ==========================================
    # WRITE REPORTS
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
        selected_feature_indices=selected_feature_indices,
        selected_feature_values=selected_feature_values,
        topk_report_path=json_topk_path.with_name(
            f"topk_feature_report_{diversity_tag}.json"
        ),
    )

    log(f"TOTAL TIME: {time.time() - t0_total:.2f}s")

    return (
        selected_feature_indices,
        selected_feature_values,
    )



def get_top_sample_indices(
    dataloader,
    model,
    sae,
    hook_name,
    layer_idx,
    feature_indices,
    n_samples,
    sequence_aggregation: Literal["sum", "mean", "max", "last_token"],
    feature_aggregation: Literal["sum", "mean"],
    device="cuda",
    chunk_size: int = 256,  # <-- Added chunk size parameter
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
    top_heap = []

    print(
        f"Scanning dataset (Heap Optimized). Aggregation: Seq='{sequence_aggregation}', Feat='{feature_aggregation}'..."
    )

    dataset_start_index = 0
    total_batches = len(dataloader) if hasattr(dataloader, "__len__") else None

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

            # 2 & 3. Encode and Filter (Chunked to save VRAM)
            flat_acts = scaled_acts.view(-1, acts.shape[-1])
            chunked_relevant_acts = []
            
            for start_idx in range(0, flat_acts.shape[0], chunk_size):
                chunk = flat_acts[start_idx : start_idx + chunk_size]
                chunk_features = sae.encode(chunk)
                
                # Filter IMMEDIATELY. This drops the massive unused feature dimension 
                # before we accumulate the chunks, saving huge amounts of VRAM.
                filtered_chunk = chunk_features[:, feature_indices]
                chunked_relevant_acts.append(filtered_chunk)
            
            # Concatenate the filtered chunks
            relevant_acts_flat = torch.cat(chunked_relevant_acts, dim=0)
            
            # Reshape back to (batch, seq_len, len(feature_indices))
            relevant_acts = relevant_acts_flat.view(acts.shape[0], acts.shape[1], -1)

            # 4. Sequence Aggregation
            if sequence_aggregation == "sum":
                seq_scores = relevant_acts.sum(dim=1)
            elif sequence_aggregation == "mean":
                seq_scores = relevant_acts.mean(dim=1)
            elif sequence_aggregation == "max":
                seq_scores = relevant_acts.max(dim=1).values
            elif sequence_aggregation == "last_token":
                seq_scores = relevant_acts[:, -1, :]
            else:
                raise ValueError("sequence_aggregation must be 'sum', 'mean', 'max', or 'last_token'")

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
                    heapq.heappush(top_heap, (score, idx))
                else:
                    heapq.heappushpop(top_heap, (score, idx))

        # Cleanup (Updated to reflect removed intermediate variables)
        del (
            tokens,
            acts,
            scaled_acts,
            flat_acts,
            chunked_relevant_acts,
            relevant_acts_flat,
            relevant_acts,
            seq_scores,
        )
        torch.cuda.empty_cache()

        dataset_start_index += batch_len

    # --- Final Sort ---
    top_heap.sort(key=lambda x: x[0], reverse=True)

    return [idx for score, idx in top_heap]


def convert_alpaca_to_messages(examples, is_inference=False):
    """Converts Alpaca instruction/input/output format into standard messages format."""
    # Safely get columns, defaulting to empty strings if missing
    length = len(examples.get("output", examples.get("instruction", [])))
    instructions = examples.get("instruction", [""] * length)
    inputs = examples.get("input", [""] * length)
    outputs = examples.get("output", [""] * length)
    
    batch_conversations = []
    
    for inst, inp, out in zip(instructions, inputs, outputs):
        inst_stripped = inst.strip()
        inp_stripped = inp.strip()
        
        # Combine instruction and input cleanly
        if inst_stripped and inp_stripped:
            user_content = f"{inst_stripped}\n\n{inp_stripped}"
        elif inp_stripped:
            user_content = inp_stripped
        else:
            user_content = inst_stripped
            
        conversation = [{"role": "user", "content": user_content}]
        
        # Append assistant response if we are training
        if not is_inference and out:
            conversation.append({"role": "assistant", "content": out.strip()})
            
        batch_conversations.append(conversation)
        
    return {"messages": batch_conversations}


def format_alpaca(tokenizer, examples, is_inference=False):
    """Wraps the conversion and formatting processes."""
    # 1. Convert to standard messages format
    standardized_examples = convert_alpaca_to_messages(examples, is_inference)
    
    # 2. Pass the converted messages to the existing formatter
    return format_standard_messages(tokenizer, standardized_examples, is_inference)


def format_standard_messages(tokenizer, examples, is_inference=False):
    return {'messages': examples['messages']}
    # has_prompt = "prompt" in examples
    # has_instruction = "instruction" in examples
    
    # batch_conversations = []
    
    # for i, messages in enumerate(examples["messages"]):
    #     # Create a shallow copy of the message list
    #     prompt_messages = list(messages)
        
    #     # Gather anything that needs to be prepended
    #     prepends = []
    #     if has_instruction and examples["instruction"][i]:
    #         prepends.append(examples["instruction"][i].strip())
    #     if has_prompt and examples["prompt"][i]:
    #         prepends.append(examples["prompt"][i].strip())
            
    #     # Prepend to the first user message
    #     if prepends and prompt_messages and prompt_messages[0]["role"] == "user":
    #         combined_prepend = "\n\n".join(prepends)
    #         current_content = prompt_messages[0]["content"]
            
    #         # Safeguard: Don't prepend if the text is already identical
    #         if combined_prepend not in current_content:
    #             # Create a new dict for the modified message to avoid mutating original
    #             first_msg = prompt_messages[0].copy()
    #             first_msg["content"] = f"{combined_prepend}\n\n{current_content}"
    #             prompt_messages[0] = first_msg
        
    #     # Strip final assistant turn if doing inference
    #     if is_inference and prompt_messages and prompt_messages[-1]["role"] == "assistant":
    #         prompt_messages = prompt_messages[:-1]
            
    #     batch_conversations.append(prompt_messages)
            
    # # Apply template to the ENTIRE batch at once
    # formatted_texts = tokenizer.apply_chat_template(
    #     batch_conversations, 
    #     tokenize=False, 
    #     add_generation_prompt=is_inference
    # )
        
    # return {"text": formatted_texts}


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
    model_label = suite.dirs.get_trained_model_root(dataset_size=dataset_size, topk=topk).name
    print(f"[PRUNE] Starting pruning for model {model_label} (dataset_size={dataset_size}, topk={topk})...")

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

    print(f"--- Prune: model={model_label} ---")
    # tokenizer = AutoTokenizer.from_pretrained(suite.sft.cfg.tokenizer_id, fix_mistral_regex=True)
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # tokenizer.padding_side = 'left'  # for inference
    model = HookedTransformer.from_pretrained_no_processing(
        model_id, device=device, dtype=torch.bfloat16
    )
    # model, tokenizer = ensure_pad_token(model=model, tokenizer=tokenizer)

    model, tokenizer = ensure_pad_token_hooked_transformer(model=model, is_inference=True)
    chat_tokenizer = AutoTokenizer.from_pretrained(suite.sft.cfg.tokenizer_id)

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
    chat_str_dataset = str_dataset.map(
        lambda batch: {
            "text": [
                chat_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False) 
                for messages in batch["messages"]
            ]
        },
        batched=True,
        batch_size=1000,
        num_proc=12,
    )


    tokenized_dataset = chat_str_dataset.map(
        lambda examples: tokenizer(
            examples["text"],
            # examples["messages"],
            padding="max_length",
            truncation=True,
            max_length=suite.sft.cfg.max_seq_length,
            return_tensors=None 
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
        diversity_param=suite.prune.cfg.diversity_param,
        diversity_seed=suite.seed,
        diversity_tag=suite.prune.cfg.diversity_tag,
        full_report_path_csv=suite.dirs.get_dataset_path(
            dataset_size=dataset_size, topk=topk
        ).parent / "full_feature_report.csv",
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
        "seed": suite.active_seed,
    }
    with open(dataset_cfg_path, "w") as f:
        json.dump(model_cfg, f, indent=4)

    print(f"[PRUNE] Dataset generated for model {model_label}.")
