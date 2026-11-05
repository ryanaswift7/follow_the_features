#!/bin/bash

uv run ftf -c configs/ordering/llama_ultrachat.yaml
uv run ftf -c configs/ordering/olmo_dolci.yaml
uv run ftf -c configs/ordering/qwen_alpaca.yaml
