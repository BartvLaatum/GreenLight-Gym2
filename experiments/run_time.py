import argparse
from gl_gym.environments.tomato_env import TomatoEnv
from gl_gym.common.utils import load_env_params
import numpy as np
import pandas as pd
import time
import timeit


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
    env_base_params["dt"] = 300

    # Initialize the environment with both parameter dictionaries
    env = TomatoEnv(base_env_params=env_base_params, **env_specific_params)
    crop_DM = 6240*10
    controls = pd.read_csv('data/bleiswijk/controls2009.csv').values

    def run_simulation():
        env.reset(seed=env_seed)
        env.set_crop_state(cBuf=0, cLeaf=0.7*crop_DM, cStem=0.25*crop_DM, cFruit=0.05*crop_DM, tCanSum=0)
        done = False
        time_start = time.time()
        while not done:
            x, done = env.step_raw_control(controls[env.timestep])
        time_end = time.time()
        return time_end - time_start


    # Time the execution of the function
    elapsed_times = []
    for i in range(10):
        elapsed_time = run_simulation()
        elapsed_times.append(elapsed_time)
    # save elapsed times to csv
    df = pd.DataFrame(elapsed_times, columns=["elapsed_time"])
    df.to_csv("data/bleiswijk/elapsed_times.csv", index=False)
    print(f"Elapsed time: {np.mean(elapsed_times):.4f} seconds")  # Print elapsed time
