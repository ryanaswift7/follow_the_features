import torch
import tqdm
from datasets import load_dataset
from sae_lens import SAE
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer

# ================= CONFIGURATION =================
MODEL_ID = "google/gemma-2-9b-it"
SAE_RELEASE = "gemma-scope-9b-it-res"
SAE_ID = "layer_20/width_16k/average_l0_91"
DATASET_ID = "yahma/alpaca-cleaned"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
MAX_LENGTH = 5000
TOP_K_FEATURES = 20
PRUNE_THRESHOLD = 25.0
NUM_SAMPLES = 1000
# =================================================


def main():
    print("--- Configuration ---")
    print(f"Model: {MODEL_ID}")
    print(f"Max Sequence Length: {MAX_LENGTH}")
    print(f"Sample Limit: {NUM_SAMPLES if NUM_SAMPLES else 'ALL'}")
    print(f"Device: {DEVICE}")

    # 1. Load Tokenizer
    print("\n[1/5] Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # 2. Load Model & SAE
    print("[2/5] Loading Model & SAE...")
    model = HookedTransformer.from_pretrained_no_processing(
        MODEL_ID, device=DEVICE, dtype=torch.bfloat16
    )
    sae, _, _ = SAE.from_pretrained_with_cfg_and_sparsity(
        release=SAE_RELEASE, sae_id=SAE_ID, device=DEVICE
    )

    # 3. Load & Slice Dataset
    print(f"[3/5] Loading Dataset '{DATASET_ID}'...")
    dataset = load_dataset(DATASET_ID, split="train")

    if NUM_SAMPLES is not None:
        if NUM_SAMPLES < len(dataset):
            print(f"      > Slicing dataset to first {NUM_SAMPLES} samples.")
            dataset = dataset.select(range(NUM_SAMPLES))
        else:
            print(
                f"      > NUM_SAMPLES ({NUM_SAMPLES}) is larger than dataset. Using full dataset."
            )

    # 4. Pre-process (Chat Template)
    print("\n[4/5] Formatting dataset with Chat Template (via map)...")

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

    dataset = dataset.map(apply_chat_formatting, batched=True, batch_size=1000)

    # # --- NEW: Filter by Sequence Length ---
    # print(f"\n[4.5/5] Filtering samples longer than {MAX_LENGTH} tokens...")

    # # We use the model's tokenizer to measure length accurately
    # def filter_long_sequences(example):
    #     # We only need the length, so we don't return the tokens to save RAM
    #     # HookedTransformer's tokenizer is accessible via model.tokenizer
    #     tokens = model.tokenizer(
    #         example["formatted_text"], truncation=False, add_special_tokens=True
    #     )["input_ids"]
    #     return len(tokens) <= MAX_LENGTH

    # original_count = len(dataset)
    # dataset = dataset.filter(filter_long_sequences)
    # new_count = len(dataset)

    # print(f"      > Dropped {original_count - new_count} samples exceeding limit.")
    # print(f"      > Remaining count: {new_count}")

    # 5. Run Inference & Prune
    print("\n[5/5] Running Inference & Pruning...")

    feature_activity = torch.zeros(sae.cfg.d_sae, device="cpu")
    hook_name = sae.cfg.metadata.hook_name
    layer_idx = int(hook_name.split(".")[1])

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)

    # --- Pass 1: Identify Top Features ---
    print("      > Pass 1: identifying top features...")
    for batch in tqdm.tqdm(dataloader, desc="Scanning"):
        # Note: We can now safely set truncate=False or keep it as safeguard
        tokens = model.to_tokens(
            batch["formatted_text"],
            truncate=True,  # Safe now, but good practice to keep
        ).to(DEVICE)

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

    top_values, top_indices = torch.topk(feature_activity, TOP_K_FEATURES)
    top_indices = top_indices.cpu().tolist()
    print(f"      > Top {TOP_K_FEATURES} Feature Indices: {top_indices}")

    # --- Pass 2: Select Samples ---
    print("      > Pass 2: Selecting samples matching top features...")

    kept_indices = []

    for i, batch in tqdm.tqdm(
        enumerate(dataloader), total=len(dataloader), desc="Pruning"
    ):
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

                if relevant_features.mean().item() > PRUNE_THRESHOLD:
                    global_idx = i * BATCH_SIZE + local_idx
                    kept_indices.append(global_idx)

            del (
                logits,
                cache,
                acts,
                sample_acts,
                sample_feature_acts,
                relevant_features,
            )
            torch.cuda.empty_cache()

    # Final Selection
    if kept_indices:
        # Note: 'dataset' here is already the filtered version
        pruned_dataset = dataset.select(kept_indices)
        save_name = f"alpaca_gemma2_top{TOP_K_FEATURES}_subset.hf"
        pruned_dataset.save_to_disk(save_name)

        print("\n--- SUCCESS ---")
        print(f"Input Count (Length Filtered): {len(dataset)}")
        print(f"Pruned Count (SAE Filtered):   {len(pruned_dataset)}")
        print(f"Saved to:       {save_name}")
    else:
        print("\n--- RESULT ---")
        print("No samples met the activation threshold.")


if __name__ == "__main__":
    main()
