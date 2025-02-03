import argparse
from gl_gym.environments.tomato_env import TomatoEnv
from gl_gym.common.utils import load_env_params
import numpy as np
import pandas as pd
import time

def interpolate_weather_data(weather, env_base_params):
    """
    Interpolates weather data to match the simulation timestep.

    Args:
        weather (np.ndarray): Original weather data.
        env_base_params (dict): Environment base parameters containing 'dt' and 'season_length'.

    Returns:
        np.ndarray: Interpolated weather data.
    """
    timesteps_per_hour = int(3600 / env_base_params['dt'])
    weather_timesteps = len(weather)
    target_timesteps = timesteps_per_hour * 24 * env_base_params['season_length'] + 1

    # Create time arrays for interpolation
    t_orig = np.linspace(0, env_base_params['season_length'], weather_timesteps) 
    t_interp = np.linspace(0, env_base_params['season_length'], target_timesteps)

    # Interpolate each weather variable
    weather_interp = np.zeros((target_timesteps, weather.shape[1]))
    for i in range(weather.shape[1]):
        weather_interp[:, i] = np.interp(t_interp, t_orig, weather[:, i])

    return weather_interp



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
    env_base_params["season_length"] = 5
    env_base_params["nu"] = 6
    env_base_params["dt"] = 300
    env_base_params["pred_horizon"] = 0

    dt_data = 300
    Ns_data = int(86400*env_base_params['season_length']/dt_data) + 1

    # Initialize the environment with both parameter dictionaries
    env = TomatoEnv(base_env_params=env_base_params, **env_specific_params)
    crop_DM = 6240*10
    controls = pd.read_csv('data/comparison/controls2009.csv').values[:Ns_data, :6]
    weather = pd.read_csv('data/comparison/weather2009.csv').values[:Ns_data, :10]
    weather = interpolate_weather_data(weather, env_base_params)

    def run_simulation():
        print(env.N)
        env.reset(seed=env_seed)
        env.weather_data = weather
        # env.set_crop_state(cBuf=0, cLeaf=0.7*crop_DM, cStem=0.25*crop_DM, cFruit=0.05*crop_DM, tCanSum=0)

        done = False
        time_start = time.time()
        Xs = [np.copy(env.x)]
        Us = [np.copy(env.u)]
        print(env.x[0])
        while not done:
            x, done = env.step_raw_control(controls[env.timestep])
            Xs.append(x)
            Us.append(env.u)
        Xs = np.array(Xs)
        Us = np.array(Us)
        return Xs, Us

    # Time the execution of the function
    Xs, Us = run_simulation()
    # save elapsed times to csv
    state_columns = ["co2Air", "co2Top", "tAir", "tTop", "tCan", "tCovIn", "tCovE",
                      "tThScr", "tFlr", "tPipe", "tSo1", "tSo2", "tSo3", "tSo4", "tSo5", 
                      "vpAir", "vpTop", "tLamp", "tIntLamp", "tGroPipe", "tBlScr", "tCan24",
                      "cBuf", "cLeaf", "cStem", "cFruit", "tCanSum", "time"]
    print(np.array(Xs).shape)
    states = pd.DataFrame(np.array(Xs), columns=state_columns)
    states.to_csv("data/comparison/states_pipeinput.csv", index=False)

    weather_cols = ["glob_rad", "temp", "vpout", "co2out", "wind", "tsky", "tso", "dli", "isday", "isday_smooth"]   
    controls_cols = ["uBoil", "uCO2", "uThScr", "uVent", "uLamp", "uBlScr"]
    weather_data = pd.DataFrame(env.weather_data, columns=weather_cols)
    weather_data.to_csv("data/comparison/weather_pipeinput.csv", index=False)
    controls_data = pd.DataFrame(controls, columns=controls_cols)
    controls_data.to_csv("data/comparison/controls_pipeinput.csv", index=False)