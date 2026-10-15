#!/bin/bash

uv run fdp -c configs/ordering/llama_ultrachat.yaml
uv run fdp -c configs/ordering/olmo_dolci.yaml
uv run fdp -c configs/ordering/qwen_alpaca.yaml