def run_prune(config, top_k, threshold, pruned_path, dataset_dir):
    from mvp.utils.experiment_helpers import ensure_dir

    print(f"[PRUNE] Starting pruning for top_k={top_k}, threshold={threshold}")
    ensure_dir(dataset_dir)
    prune_run(
        config=config,
        top_k_features=top_k,
        prune_threshold=threshold,
        output_path=pruned_path,
    )
    print(f"[PRUNE] Dataset generated for topk{top_k}_thresh{threshold}.")


def prune_run(config, top_k_features, prune_threshold, output_path):
    import torch
    from datasets import load_dataset
    from sae_lens import SAE
    from transformer_lens import HookedTransformer
    from transformers import AutoTokenizer

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Unpack config
    model_id = config.prune.model_id
    sae_release = config.prune.sae_release
    sae_id = config.prune.sae_id
    dataset_id = config.prune.dataset_id
    batch_size = config.prune.batch_size
    max_length = config.prune.max_length
    num_samples = config.prune.num_samples
    batch_size = config.prune.batch_size

    print(f"--- Prune: top_k={top_k_features}, threshold={prune_threshold} ---")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_id)
    model = HookedTransformer.from_pretrained_no_processing(
        model_id, device=DEVICE, dtype=torch.bfloat16
    )
    sae, _, _ = SAE.from_pretrained_with_cfg_and_sparsity(
        release=sae_release, sae_id=sae_id, device=DEVICE
    )
    dataset = load_dataset(dataset_id, split="train")
    if num_samples is not None and num_samples < len(dataset):
        dataset = dataset.select(range(num_samples))

    def apply_chat_formatting(examples):
        formatted_texts = []
        for inst, inp in zip(examples["instruction"], examples["input"]):
            conversation = [
                {"role": "user", "content": f"{inst}\n{inp}" if inp else inst}
            ]
            text = tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
            formatted_texts.append(text)
        return {"formatted_text": formatted_texts}

    dataset = dataset.map(apply_chat_formatting, batched=True, batch_size=batch_size)

    feature_activity = torch.zeros(sae.cfg.d_sae, device="cpu")
    hook_name = sae.cfg.metadata.hook_name
    layer_idx = int(hook_name.split(".")[1])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    for batch in dataloader:
        tokens = model.to_tokens(batch["formatted_text"], truncate=True).to(DEVICE)
        with torch.no_grad():
            logits, cache = model.run_with_cache(
                tokens, names_filter=[hook_name], stop_at_layer=layer_idx + 1
            )
            acts = cache[hook_name]
            flat_acts = acts.reshape(-1, acts.shape[-1])
            feature_acts = sae.encode(flat_acts)
            feature_activity += feature_acts.mean(dim=0).cpu()
            del logits, cache, acts, flat_acts, feature_acts
            torch.cuda.empty_cache()
    top_values, top_indices = torch.topk(feature_activity, top_k_features)
    top_indices = top_indices.cpu().tolist()
    kept_indices = []
    for i, batch in enumerate(dataloader):
        tokens = model.to_tokens(batch["formatted_text"], truncate=True).to(DEVICE)
        with torch.no_grad():
            logits, cache = model.run_with_cache(
                tokens, names_filter=[hook_name], stop_at_layer=layer_idx + 1
            )
            acts = cache[hook_name]
            for local_idx in range(len(batch["instruction"])):
                sample_acts = acts[local_idx]
                sample_feature_acts = sae.encode(sample_acts)
                relevant_features = sample_feature_acts[:, top_indices]
                if relevant_features.mean().item() > prune_threshold:
                    global_idx = i * batch_size + local_idx
                    kept_indices.append(global_idx)
            del logits, cache, acts, sample_acts, sample_feature_acts, relevant_features
            torch.cuda.empty_cache()
    if kept_indices:
        pruned_dataset = dataset.select(kept_indices)
        pruned_dataset.save_to_disk(output_path)
    else:
        print("No samples met the activation threshold.")
    del model, sae, tokenizer
    torch.cuda.empty_cache()
