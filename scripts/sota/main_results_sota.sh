#!/bin/bash
set -euo pipefail

uv run grape-s2l \
  -c configs/filtering/llama_ultrachat_100_1k.yaml \
  -s configs/sota/llama_ultrachat_100_1k.yaml

uv run grape-s2l \
  -c configs/filtering/qwen_alpaca_100_1k.yaml \
  -s configs/sota/qwen_alpaca_100_1k.yaml
