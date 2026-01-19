import torch
from datasets import load_from_disk
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer

# ================= CONFIGURATION =================
MODEL_ID = "google/gemma-3-1b-pt"
TOKENIZER_ID = "google/gemma-3-1b-it"

DATASET_PATH = "alpaca_gemma3_top10_subset.hf"
OUTPUT_DIR = "gemma3-1b-alpaca-lora-4k-modern"

# Hyperparameters
MAX_SEQ_LENGTH = 4096
LEARNING_RATE = 2e-4
# =================================================


def main():
    print(f"Loading Dataset from {DATASET_PATH}...")
    dataset = load_from_disk(DATASET_PATH)

    # 1. Chat Formatting (Standard Messages Format)
    # TRL expects a list of dictionaries with "role" and "content"
    def format_to_conversation(sample):
        user_prompt = sample["instruction"]
        if sample["input"]:
            user_prompt += f"\n{sample['input']}"

        return {
            "messages": [
                {"role": "user", "content": user_prompt},
                {"role": "model", "content": sample["output"]},
            ]
        }

    print("Formatting dataset to 'messages'...")
    dataset = dataset.map(format_to_conversation, remove_columns=dataset.column_names)

    # 2. Tokenizer
    print(f"Loading Tokenizer: {TOKENIZER_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    tokenizer.padding_side = "right"

    # 3. Model Loading (High Performance BF16)
    print(f"Loading Model: {MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
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

    # 5. Training Configuration (Replaces DataCollator)
    # - assistant_only_loss=True: Replaces DataCollatorForCompletionOnlyLM
    # - dataset_kwargs: Ensures TRL doesn't try to add special tokens twice
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        # --- The New Masking Method ---
        dataset_text_field="messages",  # Tells TRL to look for the 'messages' column
        assistant_only_loss=True,  # Automatically masks User turns (loss=0)
        max_length=MAX_SEQ_LENGTH,
        # Optimization for 192GB VRAM
        num_train_epochs=1,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        gradient_checkpointing=False,  # Speed up training (uses more VRAM)
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        # Precision
        fp16=False,
        bf16=True,
        # Logging
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
    )

    print("Starting Training (Modern TRL Workflow)...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        # Note: No data_collator passed here; SFTTrainer handles it internally
    )

    trainer.train()

    print(f"Saving Adapter to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()
