#!/bin/bash


# Array of uncertainty scales to test
uncertainty_scales=(0.0 0.05 0.1 0.15 0.2 0.25 0.3)

# Loop through each model and uncertainty scale
for i in "${!uncertainty_scales[@]}"; do
    scale="${uncertainty_scales[$i]}"
    echo "Evaluating rule-based baseline with uncertainty scale: $scale"
    python gl_gym/experiments/evaluate_baseline.py --project AgriControl --env_id TomatoEnv --mode stochastic --uncertainty_scale "$scale" 
done

echo "All evaluations completed!"
