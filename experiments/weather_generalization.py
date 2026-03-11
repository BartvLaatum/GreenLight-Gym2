import argparse

import numpy as np

from RL.experiment_manager import ExperimentManager
from RL.utils import load_env_params, load_model_hyperparams

def main(args):
    # Initialize the experiment manager
    env_config_path = f"gl_gym/configs/envs/"
    env_kwargs = load_env_params(args.env_id, env_config_path)


    # for uncertainty in uncertainties:
    hyperparameters = load_model_hyperparams(args.algorithm, args.env_id)

    experiment_manager = ExperimentManager(
        env_id=args.env_id,
        project=args.project,
        # env_base_params=env_base_params,
        env_specific_params=env_kwargs,
        hyperparameters=hyperparameters,
        group="-".join([loc[:3] for loc in args.train_locations]) + f"-{args.algorithm}",
        n_eval_episodes=args.n_eval_episodes,
        n_evals=args.n_evals,
        algorithm=args.algorithm,
        env_seed=args.env_seed,
        model_seed=args.model_seed,
        stochastic=False,
        save_model=args.save_model,
        save_env=args.save_env,
        device=args.device
    )

    # Get the current train_years attribute from the vectorized environment
    current_train_years = experiment_manager.env.get_attr("train_years")
    new_years = list(range(2014, 2021))

    # Update the train_years attribute for each environment in the vectorized env
    for i in range(len(current_train_years)):
        updated_years = current_train_years[i] + new_years
        experiment_manager.env.set_attr("train_years", updated_years, indices=i)
    experiment_manager.run_experiment()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="AgriControl", help="Wandb project name")
    parser.add_argument("--env_id", type=str, default="GreenLightEnv", help="Environment ID")
    parser.add_argument("--algorithm", type=str, default="ppo", help="RL algorithm to use")
    parser.add_argument("--group", type=str, default="group1", help="Wandb group name")
    parser.add_argument("--n_eval_episodes", type=int, default=1, help="Number of episodes to evaluate the agent for")
    parser.add_argument("--n_evals", type=int, default=10, help="Number times we evaluate algorithm during training")
    parser.add_argument("--env_seed", type=int, default=666, help="Random seed for the environment for reproducibility")
    parser.add_argument("--model_seed", type=int, default=666, help="Random seed for the RL-model for reproducibility")
    parser.add_argument("--test_location", type=str, default="Amsterdam", help="The location to test the model on")
    parser.add_argument("--train_locations", nargs='+', default=["Amsterdam"], help="The locations to train the model on")
    parser.add_argument("--device", type=str, default="cpu", help="The device to run the experiment on")
    parser.add_argument("--save_model", default=True, action=argparse.BooleanOptionalAction, help="Whether to save the model")
    parser.add_argument("--save_env", default=True, action=argparse.BooleanOptionalAction, help="Whether to save the environment")
    args = parser.parse_args()
    main(args)
