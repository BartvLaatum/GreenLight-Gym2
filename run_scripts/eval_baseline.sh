#!/bin/bash

# Loop through each model and uncertainty scale
python experiments/evaluate_baseline.py
    --env_id GreenLightEnv
    --env_config gl_gym/configs/envs/
    --rb_config configs/agents/
    --scenarios '[{"location":"Amsterdam","growth_year":2010,"start_day":59}]'
    --n_sims 1
    --save_dir results/baseline/
