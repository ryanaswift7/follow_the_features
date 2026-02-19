import gc
import json
import os
import shutil
from pathlib import Path
from typing import Optional, Union

import torch
from datasets import load_from_disk
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer

from src.schemas.suite import SuiteConfig
from src.utils.constants import MODEL_CONFIG_FILENAME, PRUNE_ID_KEY, SFT_ID_KEY

PEFT_CONFIG = LoraConfig(
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


def generate_sft_trainer_args(
    suite: SuiteConfig, output_dir: Optional[Union[str, Path]]
):
    max_seq_length = suite.sft.cfg.max_seq_length
    learning_rate = suite.sft.cfg.learning_rate
    batch_size = suite.sft.params.batch_size
    seed = suite.seed

    return SFTConfig(
        output_dir=output_dir,
        dataset_text_field="messages",
        max_length=max_seq_length,
        num_train_epochs=1,
        per_device_train_batch_size=batch_size,
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


def run_sft(suite: SuiteConfig, dataset_size: int, topk: int):
    print(f"[SFT] Starting SFT for dataset_size={dataset_size}, topk={topk}...")

    model_id = suite.sft.cfg.base_model_id
    tokenizer_id = suite.sft.cfg.tokenizer_id
    dataset_path = suite.dirs.get_dataset_path(dataset_size, topk)
    output_dir = suite.dirs.get_trained_model_root(dataset_size, topk)

    dataset = load_from_disk(dataset_path)
    dataset = dataset.map(format_to_conversation, remove_columns=dataset.column_names)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.bfloat16
    )

    trainer = SFTTrainer(
        model=model,
        args=generate_sft_trainer_args(suite=suite, output_dir=output_dir),
        train_dataset=dataset,
        peft_config=PEFT_CONFIG,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(output_dir)

    model_cfg_path = os.path.join(output_dir, MODEL_CONFIG_FILENAME)
    model_cfg = {
        PRUNE_ID_KEY: suite.ids.prune,
        SFT_ID_KEY: suite.ids.sft,
    }
    with open(model_cfg_path, "w") as f:
        json.dump(model_cfg, f, indent=4)

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
