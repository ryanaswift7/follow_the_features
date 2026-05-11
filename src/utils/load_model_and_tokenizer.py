from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from src.utils.ensure_pad_token import ensure_pad_token

def load_model_and_tokenizer(model_id: str, tokenizer_id: str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, fix_mistral_regex=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.bfloat16
    )
    model, tokenizer = ensure_pad_token(model=model, tokenizer=tokenizer)
    return model, tokenizer