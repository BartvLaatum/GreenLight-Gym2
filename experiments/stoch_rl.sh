#!/bin/bash

# Array of model names to evaluate
models=("divine-glitter-278" "resilient-monkey-279" "crimson-bird-280" "glad-haze-281" "zany-meadow-282" "true-yogurt-283" "giddy-disco-284")

# Array of uncertainty scales to test
uncertainty_scales=(0.0 0.05 0.1 0.15 0.2 0.25 0.3)
# Loop through each model and uncertainty scale
for i in "${!models[@]}"; do
    model="${models[$i]}"
    scale="${uncertainty_scales[$i]}"
    echo "Evaluating model: $model with uncertainty scale: $scale"
    python experiments/evaluate_rl.py --project AgriControl --env_id TomatoEnv --model "$model" --mode stochastic --uncertainty_scale "$scale"
done

echo "All evaluations completed!"