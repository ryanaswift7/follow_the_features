#!/bin/bash
set -euo pipefail

bash scripts/filtering/ablations_superfilter_feat_corr.sh
bash scripts/filtering/lr_agg_full_ft.sh
