#!/bin/bash

# MODELS FOR STOCHASTIC PPO
# models=("hopeful-wind-295" "light-wave-296" "ruby-star-297" "eager-resonance-298" "rural-eon-300" "stellar-durian-301" "copper-dawn-303")

# MODELS FOR STOCHASTIC SAC
models=("distinctive-frost-299" "stoic-moon-302" "graceful-dream-304" "copper-frog-305" "warm-flower-306" "sunny-sky-307" "leafy-cloud-308")

# Array of uncertainty scales to test
uncertainty_scales=(0.0 0.05 0.1 0.15 0.2 0.25 0.3)

# Loop through each model and uncertainty scale
for i in "${!models[@]}"; do
    model="${models[$i]}"
    scale="${uncertainty_scales[$i]}"
    echo "Evaluating model: $model with uncertainty scale: $scale"
    python experiments/evaluate_rl.py --project AgriControl --env_id TomatoEnv --model "$model" --mode stochastic --uncertainty_scale "$scale" --algorithm sac
done

echo "All evaluations completed!"
