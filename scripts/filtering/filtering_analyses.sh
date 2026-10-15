#!/bin/bash

uv run fdp -c configs/filtering/filter_analysis/mean_seq_agg.yaml
uv run fdp -c configs/filtering/filter_analysis/no_diversity_resample.yaml
uv run fdp -c configs/filtering/filter_analysis/no_entropy.yaml
uv run fdp -c configs/filtering/filter_analysis/superfiltering.yaml