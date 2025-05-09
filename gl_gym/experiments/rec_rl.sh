#!/bin/bash
export PYTHONPATH=$(pwd)
echo "PYTHONPATH set to: $PYTHONPATH"

# Navigate to the experiment manager directory
cd /home/bart/Documents/phd-code-projects/GL-Gym2.0

# Run the experiment manager with command line arguments
python RL/experiment_manager.py \
    --project GenerlizedWeather \
    --env_id TomatoEnv \
    --algorithm recurrentppo \
    --group rec-ppo-det \
    --n_eval_episodes 1 \
    --n_evals 1 \
    --env_seed 666 \
    --model_seed 666\
    --device cpu \
    --save_model \
    --save_env \
    --hyperparameter_tuning
