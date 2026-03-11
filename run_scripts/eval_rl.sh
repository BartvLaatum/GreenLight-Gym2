#!/bin/bash

# MODELS FOR STOCHASTIC PPO
# models=("hopeful-wind-295" "light-wave-296" "ruby-star-297" "eager-resonance-298" "rural-eon-300" "stellar-durian-301" "copper-dawn-303")

# MODELS FOR STOCHASTIC SAC
# models=("distinctive-frost-299" "stoic-moon-302" "graceful-dream-304" "copper-frog-305" "warm-flower-306" "sunny-sky-307" "leafy-cloud-308")

# Array of uncertainty scales to test
python experiments/evaluate_rl.py
    --env_id GreenLightEnv
    --env_config gl_gym/configs/envs/
    --algorithm ppo
    --model_name hopeful-wind-295
    --load_path train_data/AgriControl/ppo/stochastic/models/hopeful-wind-295
    --scenarios '[{"location":"Amsterdam","growth_year":2010,"start_day":59}]'
    --n_sims 1
    --save_dir results/rl/
