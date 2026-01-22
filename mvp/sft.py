def run_sft(config, pruned_path, model_dir, top_k, threshold):
    from mvp.utils.experiment_helpers import ensure_dir

    print(f"[SFT] Starting SFT for top_k={top_k}, threshold={threshold}")
    ensure_dir(model_dir)
    sft_run(
        config=config,
        dataset_path=pruned_path,
        output_dir=model_dir,
    )
    print(f"[SFT] Model trained for topk{top_k}_thresh{threshold}.")


import gc

import torch
from datasets import load_from_disk
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer


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


def sft_run(config, dataset_path, output_dir):
    # Use config.sft for params
    model_id = config.base_model_id
    tokenizer_id = config.tokenizer_id
    max_seq_length = config.sft.max_seq_length
    learning_rate = config.sft.learning_rate
    seed = config.seed
    # Load dataset
    dataset = load_from_disk(dataset_path)
    dataset = dataset.map(format_to_conversation, remove_columns=dataset.column_names)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.bfloat16
    )
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
    training_args = SFTConfig(
        output_dir=output_dir,
        dataset_text_field="messages",
        max_length=max_seq_length,
        num_train_epochs=1,
        per_device_train_batch_size=config.sft.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=learning_rate,
        weight_decay=0.01,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        seed=seed,
        data_seed=seed,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Only remove checkpoint directories if final model file exists
    import os
    import shutil

    final_model_path = os.path.join(output_dir, "adapter_model.safetensors")
    if os.path.isfile(final_model_path):
        for item in os.listdir(output_dir):
            if item.startswith("checkpoint-"):
                checkpoint_path = os.path.join(output_dir, item)
                shutil.rmtree(checkpoint_path)

    del model, trainer, tokenizer
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
