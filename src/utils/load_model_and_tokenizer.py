from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from src.utils.ensure_pad_token import ensure_pad_token

LLAMA_3_TRAINING_TEMPLATE = (
    "{% set loop_messages = messages %}"
    "{% for message in loop_messages %}"
        "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}"
        "{% if message['role'] == 'assistant' %}"
            "{% generation %}{{ content }}{% endgeneration %}"
        "{% else %}"
            "{{ content }}"
        "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
)

CHATML_TRAINING_TEMPLATE = (
    "{% for message in messages %}"
        "{{'<|im_start|>' + message['role'] + '\n'}}"
        "{% if message['role'] == 'assistant' %}"
            "{% generation %}{{ message['content'] + '<|im_end|>\n' }}{% endgeneration %}"
        "{% else %}"
            "{{ message['content'] + '<|im_end|>\n' }}"
        "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
        "{{ '<|im_start|>assistant\n' }}"
    "{% endif %}"
)


def load_model_and_tokenizer(model_id: str | Path, tokenizer_id: str | Path, is_inference: bool):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, fix_mistral_regex=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.bfloat16
    )
    
    # 1. Utility to set up padding
    model, tokenizer = ensure_pad_token(model=model, tokenizer=tokenizer, is_inference=is_inference)
    
    # 2. Dynamically patch the template
    model_id_lower = str(model_id).lower()
    
    if "llama-3" in model_id_lower:
        tokenizer.chat_template = LLAMA_3_TRAINING_TEMPLATE
    else:
        # Applies ChatML natively to Qwen, Olmo, or any other fallback
        tokenizer.chat_template = CHATML_TRAINING_TEMPLATE
        
    return model, tokenizer