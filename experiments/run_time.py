import argparse
from gl_gym.environments.tomato_env import TomatoEnv
from gl_gym.common.utils import load_env_params
from gl_predefined_controls import set_matlab_params
import numpy as np
import pandas as pd
import time



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="TomatoEnv")
    parser.add_argument("--env_config_path", type=str, default="gl_gym/configs/envs/")
    args = parser.parse_args()

    env_base_params, env_specific_params = load_env_params(args.env_id, args.env_config_path)
    env_seed= 666
    env_base_params["location"] = "Bleiswijk"
    env_base_params["data_source"] = "GL"
    env_base_params["start_train_year"] = 2009
    env_base_params["end_train_year"] = 2009
    env_base_params["start_train_day"] = 0
    env_base_params["end_train_day"] = 0
    env_base_params["season_length"] = 10
    env_base_params["pred_horizon"] = 0
    env_base_params["dt"] = 300

    # Initialize the environment with both parameter dictionaries
    env = TomatoEnv(base_env_params=env_base_params, **env_specific_params)
    controls = pd.read_csv('data/AgriControl/comparison/matlab/controls_pipe2009.csv').values[:, :6]
    weather = pd.read_csv('data/AgriControl/comparison/matlab/weather_pipe2009.csv').values[:, :]
    # weather = interpolate_weather_data(weather, env_base_params)
    indoor = [23.7, 1291.822759273841, 1907.926700562060]

    def run_simulation():
        print(env.N)
        env.reset(seed=env_seed)
        env.weather_data = weather
        env.p = set_matlab_params(env.p)
        env.reset(seed=env_seed)
        env.set_crop_state(cBuf=0, cLeaf=0.9e5, cStem=2.5e5, cFruit=2.8e5, tCanSum=3000)
        done = False
        time_start = time.time()
        while not done:
            obs, reward, terminated, _, _ = env.step_raw_control(controls[env.timestep])
        time_end = time.time()
        return time_end - time_start

    # Time the execution of the function
    elapsed_times = []
    for i in range(10):
        elapsed_time = run_simulation()
        elapsed_times.append(elapsed_time)
    df = pd.DataFrame(elapsed_times, columns=["elapsed_time"])
    df.to_csv("data/AgriControl/run_times/gl_gym.csv", index=False)
    print(f"Elapsed time: {np.mean(elapsed_times):.4f} seconds")  # Print elapsed time
