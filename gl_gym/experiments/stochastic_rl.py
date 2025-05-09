import argparse

import numpy as np

from RL.experiment_manager import ExperimentManager
from gl_gym.common.utils import load_model_hyperparams
from RL.utils import load_env_params

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="AgriControl", help="Wandb project name")
    parser.add_argument("--env_id", type=str, default="TomatoEnv", help="Environment ID")
    parser.add_argument("--algorithm", type=str, default="ppo", help="RL algorithm to use")
    parser.add_argument("--group", type=str, default="group1", help="Wandb group name")
    parser.add_argument("--n_eval_episodes", type=int, default=1, help="Number of episodes to evaluate the agent for")
    parser.add_argument("--n_evals", type=int, default=10, help="Number times we evaluate algorithm during training")
    parser.add_argument("--env_seed", type=int, default=666, help="Random seed for the environment for reproducibility")
    parser.add_argument("--model_seed", type=int, default=666, help="Random seed for the RL-model for reproducibility")
    parser.add_argument("--device", type=str, default="cpu", help="The device to run the experiment on")
    parser.add_argument("--save_model", default=True, action=argparse.BooleanOptionalAction, help="Whether to save the model")
    parser.add_argument("--save_env", default=True, action=argparse.BooleanOptionalAction, help="Whether to save the environment")
    args = parser.parse_args()
    env_config_path = f"gl_gym/configs/envs/"
    env_base_params, env_specific_params = load_env_params(args.env_id, env_config_path)

    # Initialize the experiment manager
    uncertainties = np.linspace(0.0, 0.3, 7)
    uncertainties.round(2)
    print(uncertainties)
    for uncertainty in uncertainties:
        hyperparameters = load_model_hyperparams(args.algorithm, args.env_id)
        group = f"{args.algorithm}-stoch-{uncertainty}"
        env_specific_params["uncertainty_scale"] = uncertainty
        experiment_manager = ExperimentManager(
            env_id=args.env_id,
            project=args.project,
            env_base_params=env_base_params,
            env_specific_params=env_specific_params,
            hyperparameters=hyperparameters,
            group=group,
            n_eval_episodes=args.n_eval_episodes,
            n_evals=args.n_evals,
            algorithm=args.algorithm,
            env_seed=args.env_seed,
            model_seed=args.model_seed,
            stochastic=True,
            save_model=args.save_model,
            save_env=args.save_env,
            device=args.device
        )
        experiment_manager.run_experiment()

