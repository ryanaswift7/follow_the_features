#!/bin/bash
set -euo pipefail

uv run ftf -c configs/filtering/filter_analysis/lr_scheduler_filter.yaml
uv run ftf -c configs/filtering/filter_analysis/mean_seq_agg.yaml

# decided not to include in paper
# uv run ftf -c configs/filtering/filter_analysis/full_ft_filter.yaml
