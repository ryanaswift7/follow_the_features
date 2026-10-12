"""Canonical SmallToLarge (S2L) data selection.

The paper's reference implementation trains a proxy, evaluates the whole
dataset at frozen checkpoints, clusters the resulting loss trajectories, and
allocates a budget across clusters in ascending cluster-size order.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import faiss
import numpy as np
import torch


def allocate_s2l_indices(
    cluster_labels: Sequence[int],
    budget: int,
    rng: np.random.Generator | None = None,
) -> list[int]:
    """Select exactly ``budget`` points using S2L Algorithm 1.

    Clusters are processed from smallest to largest.  The dynamic quota is
    ``(budget - selected) / remaining_clusters``; this is what makes S2L
    preferentially retain small clusters instead of treating every cluster as
    having the same population.
    """
    labels = np.asarray(cluster_labels)
    if labels.ndim != 1:
        raise ValueError("cluster_labels must be one-dimensional")
    if not 0 <= budget <= len(labels):
        raise ValueError(f"budget must be in [0, {len(labels)}], got {budget}")
    if budget == 0:
        return []

    rng = rng or np.random.default_rng()
    clusters = [
        np.flatnonzero(labels == label)
        for label in np.unique(labels)
    ]
    clusters.sort(key=len)

    selected: list[int] = []
    for position, cluster in enumerate(clusters):
        remaining_clusters = len(clusters) - position
        quota = (budget - len(selected)) // remaining_clusters
        if len(cluster) <= quota:
            selected.extend(int(index) for index in cluster)
        elif quota:
            chosen = rng.choice(cluster, size=quota, replace=False)
            selected.extend(int(index) for index in chosen)

    if len(selected) != budget:
        # Integer division can leave a remainder.  Fill it uniformly from the
        # not-yet-selected points, preserving the requested exact budget.
        selected_set = set(selected)
        available = np.asarray(
            [index for index in range(len(labels)) if index not in selected_set]
        )
        extra = rng.choice(available, size=budget - len(selected), replace=False)
        selected.extend(int(index) for index in extra)
    return selected


def cluster_trajectories(
    trajectories: np.ndarray | torch.Tensor,
    n_clusters: int,
    seed: int,
) -> np.ndarray:
    """Cluster raw loss trajectories with the reference FAISS procedure."""
    values = np.asarray(trajectories, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("trajectories must have shape [num_examples, num_checkpoints]")
    if len(values) == 0:
        return np.empty(0, dtype=np.int64)
    n_clusters = min(n_clusters, len(values))
    if n_clusters == 1:
        return np.zeros(len(values), dtype=np.int64)

    # The paper/reference implementation clusters the loss values directly.
    # Do not standardize columns: that changes the trajectory distance metric.
    kmeans = faiss.Kmeans(
        d=values.shape[1],
        k=n_clusters,
        niter=20,
        nredo=10,
        seed=seed,
        verbose=False,
        gpu=torch.cuda.is_available(),
    )
    kmeans.train(values)
    _, assignments = kmeans.index.search(values, 1)
    return assignments[:, 0].astype(np.int64)


def select_s2l(
    trajectories: np.ndarray | torch.Tensor,
    budget: int,
    n_clusters: int,
    seed: int,
    sources: Sequence[str] | None = None,
) -> list[int]:
    """Select a canonical S2L subset, clustering each source independently."""
    values = np.asarray(trajectories, dtype=np.float32)
    if budget > len(values):
        raise ValueError(f"S2L budget {budget} exceeds dataset size {len(values)}")
    rng = np.random.default_rng(seed)

    if sources is None:
        labels = cluster_trajectories(values, n_clusters, seed)
        return allocate_s2l_indices(labels, budget, rng)

    source_values = np.asarray(sources)
    if len(source_values) != len(values):
        raise ValueError("sources and trajectories must have the same length")

    # Source-aware clustering is performed independently.  The global
    # allocation then applies the same smallest-cluster-first rule across all
    # source clusters, as in the paper's source-wise extension.
    cluster_groups: list[np.ndarray] = []
    for source in np.unique(source_values):
        indices = np.flatnonzero(source_values == source)
        local_labels = cluster_trajectories(values[indices], n_clusters, seed)
        cluster_groups.extend(indices[local_labels == label] for label in np.unique(local_labels))

    cluster_groups.sort(key=len)
    labels = np.full(len(values), -1, dtype=np.int64)
    for cluster_id, indices in enumerate(cluster_groups):
        labels[indices] = cluster_id
    return allocate_s2l_indices(labels, budget, rng)


def collect_checkpoint_trajectories(
    model,
    tokenizer,
    dataset,
    checkpoint_paths: Sequence[str | Path],
    max_length: int,
    batch_size: int,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Evaluate every example at every frozen proxy checkpoint.

    ``model`` is reused to avoid repeatedly allocating the proxy.  Every
    trajectory column is computed from one complete frozen checkpoint, unlike
    online per-example losses collected during training.
    """
    device = device or next(model.parameters()).device
    encoded = [_encode_messages(tokenizer, item["messages"], max_length) for item in dataset]
    trajectories = torch.empty((len(encoded), len(checkpoint_paths)), dtype=torch.float32)

    for checkpoint_index, checkpoint_path in enumerate(checkpoint_paths):
        state_path = Path(checkpoint_path) / "model_state.pt"
        if not state_path.exists():
            raise FileNotFoundError(
                f"Missing frozen proxy checkpoint state: {state_path}"
            )
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
        for start in range(0, len(encoded), batch_size):
            batch = _collate(encoded[start : start + batch_size], tokenizer)
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.no_grad():
                logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                ).logits
            trajectories[start : start + len(batch["input_ids"]), checkpoint_index] = _losses(
                logits, batch["labels"]
            ).cpu()
    return trajectories


def _encode_messages(tokenizer, messages, max_length):
    full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full_ids = tokenizer(full, add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]
    prompt_ids = tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]
    labels = list(full_ids)
    for index in range(min(len(prompt_ids), len(labels))):
        labels[index] = -100
    return {"input_ids": full_ids, "labels": labels}


def _collate(examples, tokenizer):
    width = max(len(example["input_ids"]) for example in examples)
    pad = tokenizer.pad_token_id or 0
    input_ids, labels, masks = [], [], []
    for example in examples:
        padding = width - len(example["input_ids"])
        input_ids.append(example["input_ids"] + [pad] * padding)
        labels.append(example["labels"] + [-100] * padding)
        masks.append([1] * (width - padding) + [0] * padding)
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(masks),
    }


def _losses(logits, labels):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    losses = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(shift_labels.shape)
    mask = shift_labels.ne(-100)
    return (losses * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
