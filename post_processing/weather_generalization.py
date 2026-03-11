import argparse
from os import makedirs
from os.path import join

from tqdm import tqdm
import numpy as np

from stable_baselines3 import PPO, SAC

from experiments.evaluate_rl import evaluate, load_env
from common.results import Results
from RL.utils import load_env_params, load_model_hyperparams

ALG = {"ppo": PPO, 
       "sac": SAC}

MODELS = {
    "Amsterdam": "solar-morning-199",
    "Beijing": "crimson-water-200",
    "NewYork": "ruby-leaf-203",
    "London": "flowing-river-201",
    "Reykjavik": "sandy-waterfall-202",
    "ALL": "elated-thunder-206"
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="AgriControl", help="Name of the project (in wandb)")
    parser.add_argument("--env_id", type=str, default="GreenLightEnv", help="Environment ID")
    parser.add_argument("--algorithm", type=str, default="ppo", help="Name of the algorithm (ppo or sac)")
    parser.add_argument("--train_location", type=str, default="Amsterdam", help="Train location of the algorithm, for saving purposes")
    parser.add_argument("--test_location", type=str, default="Amsterdam", help="Test location of the algorithm, for evaluating")
    parser.add_argument("--uncertainty_scale", default=0., type=float, help="Uncertainty scale")
    args = parser.parse_args()

    if args.uncertainty_scale > 0:
        mode = "stochastic"
    else:
        mode = "deterministic"

    env_config_path = f"gl_gym/configs/envs/"
    load_path = f"train_data/{args.project}/{args.algorithm}/{mode}/"
    # if mode == "stochastic":
    #     save_dir = f"results/{args.project}/{mode}/{args.algorithm}-{args.train_location}/{args.uncertainty_scale}/"
    #     n_sims = 30
    # else:
    save_dir = f"results/{args.project}/{mode}/{args.algorithm}-{args.train_location}/"
    makedirs(save_dir, exist_ok=True)
    n_sims = 1 # number of simulation per year; 1 for deterministic, ~30 for stochastic
    n_years = 6
    model_name = MODELS[args.train_location]

    # load in the environment and model
    env_kwargs = load_env_params(args.env_id, env_config_path)
    model_params = load_model_hyperparams(args.algorithm, args.env_id)
    env_kwargs["uncertainty_scale"] = args.uncertainty_scale
    env_kwargs["eval_options"]["location"] = args.test_location
    eval_env = load_env(args.env_id, model_name, env_kwargs, load_path)
    # start_day = eval_env.get_attr("start_day")[0]
    # end_growth_year = eval_env.get_attr("growth_year")[0]
    location = eval_env.get_attr("location")[0]

    save_name = f"{model_name}-{location}.csv"
    print("saving results to", save_name)
