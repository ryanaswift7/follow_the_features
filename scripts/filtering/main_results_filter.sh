#!/bin/bash

# uv run ftf -c configs/filtering/llama_ultrachat_10k.yaml
uv run ftf -c configs/filtering/llama_ultrachat_100_1k.yaml

# uv run ftf -c configs/filtering/olmo_dolci_10k.yaml
uv run ftf -c configs/filtering/olmo_dolci_100_1k.yaml

# uv run ftf -c configs/filtering/qwen_alpaca_10k.yaml
uv run ftf -c configs/filtering/qwen_alpaca_100_1k.yaml
