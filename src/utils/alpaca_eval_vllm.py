from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from alpaca_eval import utils


_llm = None
_llm_model_name = None
_tokenizer = None


def vllm_local_completions(
    prompts: Sequence[str],
    model_name: str,
    max_new_tokens: int,
    is_chatml_prompt: bool = False,
    enable_thinking: bool = False,
    batch_size: int | None = None,
    model_kwargs=None,
    **decoding_kwargs,
) -> dict[str, list]:
    """Run vLLM completions with explicit control over chat-template thinking."""
    global _llm, _llm_model_name, _tokenizer

    model_kwargs = dict(model_kwargs or {})
    if batch_size is not None:
        model_kwargs["max_num_seqs"] = batch_size

    if model_name != _llm_model_name:
        logging.info("Switching vLLM model from %s to %s", _llm_model_name, model_name)
        _llm = None
        _tokenizer = None

    if _llm is None:
        logging.info("Loading vLLM model: %s, %s", model_name, model_kwargs)
        _llm = LLM(model=model_name, tokenizer=model_name, **model_kwargs)
        _llm_model_name = model_name

    if is_chatml_prompt and _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(model_name)

    sampling_params = SamplingParams(max_tokens=max_new_tokens, **decoding_kwargs)

    if is_chatml_prompt:
        prompts = [
            _tokenizer.apply_chat_template(
                utils.prompt_to_chatml(prompt),
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=enable_thinking,
            )
            for prompt in prompts
        ]

    outputs = _llm.generate(prompts, sampling_params)
    completions = [output.outputs[0].text for output in outputs]
    return {
        "completions": completions,
        "price_per_example": [np.nan] * len(completions),
        "time_per_example": [np.nan] * len(completions),
    }
