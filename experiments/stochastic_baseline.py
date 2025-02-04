import argparse
import os

from gl_gym.environments.tomato_env import TomatoEnv
from gl_gym.environments.baseline import RuleBasedController
from gl_gym.common.utils import load_env_params, load_model_hyperparams
from gl_gym.common.results import Results
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="AgriControl", help="Wandb project name")
    parser.add_argument("--env_id", type=str, default="TomatoEnv", help="Environment ID")
    args = parser.parse_args()
    save_dir = f"data/{args.project}/stochastic"
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    env_config_path = f"gl_gym/configs/envs/"
    env_base_params, env_specific_params = load_env_params(args.env_id, env_config_path)
    env_base_params['training'] = True
    rb_params = load_model_hyperparams('rule_based', args.env_id)
    rb_controller = RuleBasedController(**rb_params)
    env = TomatoEnv(base_env_params=env_base_params, **env_specific_params)
    result_columns = env.get_obs_names()[:23]
    result_columns.extend(["EPI", "Revenue", "Heat costs", "CO2 costs", "Elec costs"])
    result_columns.extend(["temp_violation", "co2_violation", "rh_violation"])
    result = Results(result_columns)

    result_data = evaluate_controller(env, rb_controller, result, save_dir)

    result.update_result(result_data)
    result.save(f"{save_dir}/rb_baseline-{env.growth_year}{env.start_day}-{env.location}.csv")
