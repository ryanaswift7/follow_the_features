#!/bin/bash

uv run fdp -c configs/filtering/filter_sweep_full_single_seed.yaml
bash scripts/filtering/filter_sweep_targeted.sh