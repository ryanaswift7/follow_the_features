import gc

import torch
from datasets import load_dataset, load_from_disk
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer

# ================= GLOBAL CONFIGURATION =================
MODEL_ID = "google/gemma-2-2b"
TOKENIZER_ID = "google/gemma-2-9b-it"

# Hyperparameters
MAX_SEQ_LENGTH = 5000
LEARNING_RATE = 2e-4
SEED = 42

MAX_ORIGINAL_SAMPLES = 1000
# ========================================================


def format_to_conversation(sample):
    user_prompt = sample["instruction"]
    if sample.get("input", ""):
        user_prompt += f"\n{sample['input']}"

    return {
        "messages": [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": sample["output"]},
        ]
    }


def train_run(dataset_path, output_dir, is_local=False, max_samples=None):
    print(f"\n{'=' * 20} STARTING RUN: {output_dir} {'=' * 20}")

    # 1. Load Dataset
    print(f"Loading Dataset from: {dataset_path}...")
    if is_local:
        dataset = load_from_disk(dataset_path)
    else:
        dataset = load_dataset(dataset_path, split="train")

    # NEW: Apply sample limit if specified
    if max_samples is not None and max_samples < len(dataset):
        print(f"Limiting dataset to {max_samples} samples...")
        # We select the first N samples to maintain reproducibility
        dataset = dataset.select(range(max_samples))

    print(f"Final dataset size: {len(dataset)}")
    print("Formatting dataset to 'messages'...")
    dataset = dataset.map(format_to_conversation, remove_columns=dataset.column_names)

    # 2. Tokenizer
    print(f"Loading Tokenizer: {TOKENIZER_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    tokenizer.padding_side = "right"

    # 3. Model Loading
    print(f"Loading Model: {MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # 4. LoRA Configuration
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    # 5. Training Configuration
    training_args = SFTConfig(
        output_dir=output_dir,
        dataset_text_field="messages",
        max_length=MAX_SEQ_LENGTH,
        num_train_epochs=1,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        seed=SEED,
        data_seed=SEED,
    )

    print(f"Starting Training for {output_dir}...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    trainer.train()

    print(f"Saving Adapter to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Cleanup
    del model
    del trainer
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return


def main():
    # --- RUN 1: Local Subset ---
    # Usually, a local subset is already small, so we pass None for max_samples
    train_run(
        dataset_path="alpaca_gemma2_top20_subset.hf",
        output_dir="gemma2-2b-alpaca-subset-lora",
        is_local=True,
        max_samples=None,
    )

    # --- RUN 2: Original Cleaned Alpaca ---
    # Here we apply the limit defined in global config
    train_run(
        dataset_path="yahma/alpaca-cleaned",
        output_dir="gemma2-2b-alpaca-cleaned-lora",
        is_local=False,
        max_samples=MAX_ORIGINAL_SAMPLES,
    )

    print("All training runs completed successfully.")


if __name__ == "__main__":
    main()
