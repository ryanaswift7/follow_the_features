#!/bin/bash

# all SAEs are medium L0 at layer 9 on Gemma 3 270m IT
# agg_max_mean is comp for 65k

configs=(
    "sae_width_16k.yaml"
    "sae_width_1m.yaml"
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