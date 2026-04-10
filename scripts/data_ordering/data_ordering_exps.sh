#!/bin/bash

# all SAEs are 65k with medium L0 at layer 9 on Gemma 3 270m IT
# agg_max_mean is comp for 65k

exps=(
    "alpaca.sh"
    "dolci.sh"
    "ultrachat.sh"
    "gemma3_suite_alpaca.sh"
)

for e in "${exps[@]}"; do
    # Assign 'e' to 'file' (or you could just use "$e" everywhere below)
    file="$e"
    
    # Changed -f (exists) to -x (executable)
    if [ -x "$file" ]; then
        echo "--- Running FDP suite in $file ---"
        # Execute the script (assuming it's in the current directory)
        ./"$file"
    elif [ -f "$file" ]; then
        echo "Warning: $file exists but is not executable. Run 'chmod +x $file'"
    else
        echo "Error: $file not found."
    fi
done