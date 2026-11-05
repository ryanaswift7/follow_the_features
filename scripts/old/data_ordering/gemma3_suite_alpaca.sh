#!/bin/bash

# all SAEs are 65k with medium L0 at layer 9 on Gemma 3 270m IT
# agg_max_mean is comp for 65k

configs=(
    "gemma3_270m_alpaca.yaml"
    "gemma3_1b_alpaca.yaml"
    "gemma3_4b_alpaca.yaml"
    "gemma3_12b_alpaca.yaml"
    "gemma3_27b_alpaca.yaml"
)

for c in "${configs[@]}"; do
    file="configs/data_ordering/${c}"
    
    if [ -f "$file" ]; then
        echo "--- Running FDP with $file ---"
        uv run ftf -c "$file"
    else
        echo "Warning: $file not found."
    fi
done
