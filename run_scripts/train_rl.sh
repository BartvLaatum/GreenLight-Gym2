#!/bin/bash

# Run the experiment manager with command line arguments
python RL/experiment_manager.py \
    --project testing \
    --env_id GreenLightEnv \
    --algorithm ppo \
    --group ppo_det \
    --n_eval_episodes 1 \
    --n_evals 10 \
    --env_seed 666 \
    --model_seed 666\
    --device cpu \
    --save_model \
    --save_env
