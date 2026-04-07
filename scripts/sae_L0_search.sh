#!/bin/bash

# both SAEs are 65k width at layer 9 on Gemma 3 270m IT
# agg_max_mean is comp for L0 medium 60

configs=(
    "L0_big_150.yaml"
    "L0_small_20.yaml"
)

for c in "${configs[@]}"; do
    file="configs/${c}"
    
    if [ -f "$file" ]; then
        echo "--- Running FDP with $file ---"
        uv run fdp -c "$file"
    else
        echo "Warning: $file not found."
    fi
done