import argparse
import os

from gl_gym.environments.tomato_env import TomatoEnv
from gl_gym.environments.baseline import RuleBasedController
from gl_gym.common.utils import load_env_params, load_model_hyperparams
from gl_gym.common.results import Results
import os
import numpy as np
from tqdm import tqdm

def evaluate_controller(env, controller, rank=0):
    epi, revenue, heat_cost, co2_cost, elec_cost = np.zeros(env.N+1), np.zeros(env.N+1), np.zeros(env.N+1), np.zeros(env.N+1),np.zeros(env.N+1)
    temp_violation, co2_violation, rh_violation = np.zeros(env.N+1), np.zeros(env.N+1), np.zeros(env.N+1)
    rewards = np.zeros(env.N+1)
    episodic_obs = np.zeros((env.N+1, 23))
    obs = env.reset(seed=666+rank)
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
    parser.add_argument("--project", type=str, default="AgriControl", help="Name of the project (in wandb)")
    parser.add_argument("--env_id", type=str, default="TomatoEnv", help="Environment ID")
    parser.add_argument("--uncertainty_scale", type=float, help="Uncertainty scale", required=True)
    parser.add_argument("--mode", type=str, choices=['deterministic', 'stochastic'], required=True)
    args = parser.parse_args()
    save_dir = f"data/{args.project}/{args.mode}/rb_baseline/"
    env_config_path = f"gl_gym/configs/envs/"
    
    if args.mode == "stochastic":
        save_dir = f"data/{args.project}/{args.mode}/rb_baseline/{args.uncertainty_scale}/"
        n_sims = 30
    else:
        save_dir = f"data/{args.project}/{args.mode}/{args.algorithm}/"
        n_sims = 1
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    rb_params = load_model_hyperparams('rule_based', args.env_id)
    rb_controller = RuleBasedController(**rb_params)

    env_base_params, env_specific_params = load_env_params(args.env_id, env_config_path)
    env_base_params['training'] = True
    eval_env = TomatoEnv(base_env_params=env_base_params, uncertainty_scale=args.uncertainty_scale, **env_specific_params)

    result_columns = eval_env.get_obs_names()[:23]
    result_columns.extend(["Rewards", "EPI", "Revenue", "Heat costs", "CO2 costs", "Elec costs"])
    result_columns.extend(["temp_violation", "co2_violation", "rh_violation"])
    result_columns.extend(["episode"])
    result = Results(result_columns)

    for sim in tqdm(range(n_sims)):
        result_data = evaluate_controller(eval_env, rb_controller, rank=sim)
        sim_column = np.full((result_data.shape[0], 1), sim)
        result_data = np.column_stack((result_data, sim_column))
        result.update_result(result_data)

        # data.append(results_data)

    start_day = eval_env.start_day
    growth_year = eval_env.growth_year
    location = eval_env.location

    save_name = f"rb_baseline-{growth_year}{start_day}-{location}.csv"
    print("saving results to", save_name)
    result.save(f"{save_dir}/{save_name}")


    # plt.figure(figsize=(10, 6))
    # for d in data:
    #     plt.plot(d[:, 5], label=f'CFruit')  # assuming cfruit is at index 7
    # plt.xlabel('Time steps')
    # plt.ylabel('CFruit')
    # plt.title('CFruit Trajectory Comparison')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

            # result.update_result(results_data)
        # result.save(f"{save_dir}/rb_baseline-{env.growth_year}{env.start_day}-{env.location}.csv")
