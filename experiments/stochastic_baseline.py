import argparse
import os

from gl_gym.environments.tomato_env import TomatoEnv
from gl_gym.environments.baseline import RuleBasedController
from gl_gym.common.utils import load_env_params, load_model_hyperparams
from gl_gym.common.results import Results
import os
from run_baseline import evaluate_controller
import matplotlib.pyplot as plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="AgriControl", help="Wandb project name")
    parser.add_argument("--env_id", type=str, default="TomatoEnv", help="Environment ID")
    args = parser.parse_args()
    save_dir = f"data/{args.project}/stochastic"
    env_config_path = f"gl_gym/configs/envs/"
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    rb_params = load_model_hyperparams('rule_based', args.env_id)
    rb_controller = RuleBasedController(**rb_params)

    result_columns = []
    result_columns.extend(["Rewards", "EPI", "Revenue", "Heat costs", "CO2 costs", "Elec costs"])
    result_columns.extend(["temp_violation", "co2_violation", "rh_violation"])
    result = Results(result_columns)
    import numpy as np

    env_base_params, env_specific_params = load_env_params(args.env_id, env_config_path)
    env_base_params['training'] = True
    uncertainties = np.linspace(0.0, 0.1, 2)

    data = []

    print(uncertainties)
    for uncertainty in uncertainties:
        print(f"Uncertainty: {uncertainty}")
        for i in range(1):
            env = TomatoEnv(base_env_params=env_base_params, uncertainty_scale=uncertainty, **env_specific_params)
            results_data = evaluate_controller(env, rb_controller, rank=i)
            data.append(results_data)
            print(results_data[:,23].sum())

    plt.figure(figsize=(10, 6))
    for d in data:
        plt.plot(d[:, 5], label=f'CFruit')  # assuming cfruit is at index 7
    plt.xlabel('Time steps')
    plt.ylabel('CFruit')
    plt.title('CFruit Trajectory Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

            # result.update_result(results_data)
        # result.save(f"{save_dir}/rb_baseline-{env.growth_year}{env.start_day}-{env.location}.csv")
