#!/bin/bash
set -euo pipefail


uv run fdp -c configs/filtering/filter_analysis/no_diversity_resample.yaml
uv run fdp -c configs/filtering/filter_analysis/no_entropy.yaml
uv run fdp -c configs/filtering/filter_analysis/superfiltering.yaml
uv run feature-rank-correlation --config configs/feature_corr/ultrachat_alpaca.yaml
