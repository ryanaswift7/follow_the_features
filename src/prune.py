import gc
import heapq
import json
import os
from typing import Literal, Optional, cast

import torch
from datasets import load_dataset
from sae_lens import SAE
from scipy.stats import differential_entropy
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer

from src.schemas.suite import SuiteConfig
from src.utils.constants import DATASET_CONFIG_FILENAME, PRUNE_ID_KEY


def estimate_vasicek_entropy(activations, window=None):
    """
    Estimates differential entropy using the Vasicek n-spacing method.
    """
    vals = activations[activations > 0].numpy()
    if len(vals) < 20:  # Minimum samples for a stable estimate
        return 1e-5
    return differential_entropy(vals)


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
    max_entropy_samples: int = 50_000,
    device="cuda",
):
    """
    Identifies the most significant SAE features using the metric:
    (Magnitude-Weighted Frequency * Decoder Norm) / Differential Entropy.

    Args:
        max_entropy_samples: Defaults to 50,000.
                             (16k features * 50k samples * 2 bytes ≈ 1.6 GB RAM)
    """

    # Initialize trackers
    d_sae = sae.cfg.d_sae
    feature_sum = torch.zeros(d_sae, device="cpu", dtype=torch.float32)

    # MEMORY FIX: Pre-allocate a bfloat16 tensor for samples
    sample_buffer = torch.zeros(
        (d_sae, max_entropy_samples), dtype=torch.bfloat16, device="cpu"
    )
    sample_counts = torch.zeros(d_sae, dtype=torch.long, device="cpu")

    total_samples_processed = 0

    print(f"Computing feature importance for {d_sae} features...")

    # Progress Bar Setup
    # Check if dataloader has a length (some streaming loaders don't)
    total_batches = len(dataloader) if hasattr(dataloader, "__len__") else None

    # Wrap dataloader with tqdm
    for i, batch in enumerate(
        tqdm(dataloader, total=total_batches, desc="Processing Batches", unit="batch")
    ):
        tokens = model.to_tokens(batch["formatted_text"], truncate=True).to(device)

        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens, names_filter=[hook_name], stop_at_layer=layer_idx + 1
            )
            acts = cache[hook_name]
            flat_acts = acts.reshape(-1, acts.shape[-1])

            # Normalize scaling
            rms = torch.sqrt(torch.mean(flat_acts**2, dim=-1, keepdim=True) + 1e-8)
            scaled_acts = flat_acts / rms

            # Encode and move to CPU immediately as bfloat16
            feature_acts = sae.encode(scaled_acts).to(
                dtype=torch.bfloat16, device="cpu"
            )

            # Accumulate Sum (cast to float32 for precision in summation)
            feature_sum += feature_acts.to(torch.float32).sum(dim=0)
            total_samples_processed += flat_acts.shape[0]

            # Collect samples for entropy
            # Find features that actually fired in this batch
            active_mask = feature_acts > 0
            unique_active_feats = torch.where(active_mask.any(dim=0))[0]

            for feat_idx in unique_active_feats:
                f_idx = feat_idx.item()
                current_count = sample_counts[f_idx]

                if current_count < max_entropy_samples:
                    # Extract nonzero values for this feature
                    vals = feature_acts[:, f_idx]
                    nonzero_vals = vals[vals > 0]

                    # Determine how many we can add
                    space_left = max_entropy_samples - current_count
                    to_add = nonzero_vals[:space_left]
                    num_to_add = to_add.size(0)

                    # Store in pre-allocated buffer
                    sample_buffer[f_idx, current_count : current_count + num_to_add] = (
                        to_add
                    )
                    sample_counts[f_idx] += num_to_add

        # Cleanup
        del tokens, acts, flat_acts, scaled_acts, feature_acts

        # Periodic Garbage Collection (every 10 batches to keep VRAM clean)
        if i % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    # --- Final Calculation ---
    print("\nCalculating Entropy and Importance Scores...")
    decoder_norms = torch.norm(sae.W_dec, p=2, dim=-1).cpu()
    final_importance = torch.zeros(d_sae)

    # Use tqdm for the final calculation as well
    for i in tqdm(range(d_sae), desc="Calculating Entropy"):
        mwf = feature_sum[i] / total_samples_processed

        # set default value for entropy
        h_clipped = 1.0

        if divide_by_entropy:
            count = sample_counts[i].item()
            if count > 0:
                # Slice only the valid samples from the buffer and cast to float32 for math
                acts_tensor = sample_buffer[i, :count].to(torch.float32)
                h = cast(float, estimate_vasicek_entropy(acts_tensor))
            else:
                h = 1.0
            h_clipped = max(h, 0.01)

        final_importance[i] = (mwf * decoder_norms[i]) / h_clipped

    top_values, top_indices = torch.topk(final_importance, k)

    return top_indices.tolist(), top_values.tolist()


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
        batch_len = len(batch["formatted_text"])
        batch_indices = range(dataset_start_index, dataset_start_index + batch_len)

        tokens = model.to_tokens(batch["formatted_text"], truncate=True).to(device)

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


def apply_chat_formatting_with_tokenizer(tokenizer, examples):
    formatted_texts = []
    for inst, inp in zip(examples["instruction"], examples["input"]):
        conversation = [{"role": "user", "content": f"{inst}\n{inp}" if inp else inst}]
        text = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        formatted_texts.append(text)
    return {"formatted_text": formatted_texts}


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
    tokenizer = AutoTokenizer.from_pretrained(suite.sft.cfg.tokenizer_id)
    model = HookedTransformer.from_pretrained_no_processing(
        model_id, device=device, dtype=torch.bfloat16
    )
    sae, _, _ = SAE.from_pretrained_with_cfg_and_sparsity(
        release=sae_release, sae_id=sae_id, device=device
    )
    dataset = load_dataset(dataset_id, split="train")
    if num_samples is not None and num_samples < len(dataset):
        dataset = dataset.select(range(num_samples))

    dataset = dataset.map(
        lambda examples: apply_chat_formatting_with_tokenizer(tokenizer, examples),
        batched=True,
        batch_size=batch_size,
    )
    feature_selection_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size
    )
    hook_name = sae.cfg.metadata.hook_name
    layer_idx = int(hook_name.split(".")[1])
    top_indices, top_values = get_topk_features(
        feature_selection_dataloader,
        model,
        sae,
        hook_name,
        layer_idx,
        k=topk,
        device=device,
        divide_by_entropy=suite.prune.cfg.divide_by_entropy,
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
