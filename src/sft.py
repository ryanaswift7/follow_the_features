import gc
import os
import shutil

import torch
from datasets import load_from_disk
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer

from src.suite_cfg_schema import SuiteConfig


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


def run_sft(suite: SuiteConfig, dataset_size: int, topk: int):
    print(f"[SFT] Starting SFT for dataset_size={dataset_size}, topk={topk}...")

    model_id = suite.sft.cfg.base_model_id
    tokenizer_id = suite.sft.cfg.tokenizer_id
    max_seq_length = suite.sft.cfg.max_seq_length
    learning_rate = suite.sft.cfg.learning_rate
    seed = suite.seed
    dataset_path = suite.dirs.get_dataset_path(dataset_size, topk)
    output_dir = suite.dirs.get_trained_model_root(dataset_size, topk)

    dataset = load_from_disk(dataset_path)
    dataset = dataset.map(format_to_conversation, remove_columns=dataset.column_names)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
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
        per_device_train_batch_size=suite.sft.params.batch_size,
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
    final_model_path = os.path.join(output_dir, "adapter_model.safetensors")
    if os.path.isfile(final_model_path):
        for item in os.listdir(output_dir):
            if item.startswith("checkpoint-"):
                checkpoint_path = os.path.join(output_dir, item)
                shutil.rmtree(checkpoint_path)

    del model, trainer, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print(
        f"[SFT] Model training finished for dataset_size={dataset_size}, topk={topk}!"
    )

    return
