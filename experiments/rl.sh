#!/bin/bash

# Navigate to the experiment manager directory
cd /home/bart/Documents/phd-code-projects/GL-Gym2.0

# # Run the experiment manager with command line arguments
# python RL/experiment_manager.py \
#     --project AgriControl \
#     --env_id TomatoEnv \
#     --algorithm ppo \
#     --group ppo_det \
#     --n_eval_episodes 1 \
#     --n_evals 10 \
#     --env_seed 666 \
#     --model_seed 666\
#     --device cpu \
#     --save_model \
#     --save_env

# Run the experiment manager with command line arguments
python RL/experiment_manager.py \
    --project AgriControl \
    --env_id TomatoEnv \
    --algorithm sac \
    --group sac_det \
    --n_eval_episodes 1 \
    --n_evals 20 \
    --env_seed 666 \
    --model_seed 666\
    --device cpu \
    --save_model \
    --save_env

