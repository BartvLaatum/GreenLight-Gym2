import argparse

import numpy as np
from gl_gym.environments.tomato_env import TomatoEnv
from gl_gym.environments.baseline import RuleBasedController
from gl_gym.common.utils import load_env_params, load_model_hyperparams
from gl_gym.common.results import Results
import os

def evaluate_controller(env, controller):
    epi, revenue, heat_cost, co2_cost, elec_cost = np.zeros(env.N+1), np.zeros(env.N+1), np.zeros(env.N+1), np.zeros(env.N+1),np.zeros(env.N+1)
    temp_violation, co2_violation, rh_violation = np.zeros(env.N+1), np.zeros(env.N+1), np.zeros(env.N+1)
    rewards = np.zeros(env.N+1)
    episodic_obs = np.zeros((env.N+1, 23))
    obs = env.reset(seed=666)
    done = False
    timestep = 0
    while not done:
        control = controller.predict(env.x, env.weather_data[env.timestep], env)

        obs, r, done, _, info = env.step_raw_control(control)
        rewards[timestep] += r
        episodic_obs[timestep] += obs[:23]
        epi[timestep] += info["EPI"]
        revenue[timestep] += info["revenue"]
        heat_cost[timestep] += info["heat_cost"]
        elec_cost[timestep] += info["elec_cost"]
        co2_cost[timestep] += info["co2_cost"]
        temp_violation[timestep] += info["temp_violation"]
        co2_violation[timestep] += info["co2_violation"]
        rh_violation[timestep] += info["rh_violation"]
        timestep += 1

    result_data = np.column_stack((episodic_obs, rewards, epi, revenue, heat_cost, co2_cost, elec_cost, temp_violation, co2_violation, rh_violation))
    return result_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="AgriControl", help="Wandb project name")
    parser.add_argument("--env_id", type=str, default="TomatoEnv", help="Environment ID")
    parser.add_argument("--stochastic", action="store_true", help="Whether to run the experiment in stochastic mode")
    args = parser.parse_args()

    if args.stochastic:
        save_dir = f"data/{args.project}/stochastic"
    else:
        save_dir = f"data/{args.project}/deterministic"
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    env_config_path = f"gl_gym/configs/envs/"
    env_base_params, env_specific_params = load_env_params(args.env_id, env_config_path)
    env_base_params['training'] = True
    rb_params = load_model_hyperparams('rule_based', args.env_id)
    rb_controller = RuleBasedController(**rb_params)
    env = TomatoEnv(base_env_params=env_base_params, **env_specific_params)
    result_columns = env.get_obs_names()[:23]
    result_columns.extend(["Rewards", "EPI", "Revenue", "Heat costs", "CO2 costs", "Elec costs"])
    result_columns.extend(["temp_violation", "co2_violation", "rh_violation"])
    result = Results(result_columns)

    result_data = evaluate_controller(env, rb_controller)

    result.update_result(result_data)
    result.save(f"{save_dir}/rb_baseline-{env.growth_year}{env.start_day}-{env.location}.csv")
