import argparse
from gl_gym.environments.tomato_env import TomatoEnv
from gl_gym.common.utils import load_env_params
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
    env_base_params["season_length"] = 5
    env_base_params["nu"] = 6
    env_base_params["dt"] = 300

    dt_data = 300
    Ns_data = int(86400*env_base_params['season_length']/dt_data)

    # Initialize the environment with both parameter dictionaries
    env = TomatoEnv(base_env_params=env_base_params, **env_specific_params)
    crop_DM = 6240*10
    controls = pd.read_csv('data/bleiswijk/controls2009.csv').values[:Ns_data, :6]
    weather = pd.read_csv('data/bleiswijk/weather2009.csv').values[:Ns_data, :10]

    # Interpolate weather data to match simulation timestep
    timesteps_per_hour = int(3600 / env_base_params['dt'])
    weather_timesteps = len(weather)
    target_timesteps = timesteps_per_hour * 24 * env_base_params['season_length']+1

    # Create time arrays for interpolation
    t_orig = np.linspace(0, env_base_params['season_length'], weather_timesteps) 
    t_interp = np.linspace(0, env_base_params['season_length'], target_timesteps)

    # Interpolate each weather variable
    weather_interp = np.zeros((target_timesteps, weather.shape[1]))
    for i in range(weather.shape[1]):
        weather_interp[:,i] = np.interp(t_interp, t_orig, weather[:,i])
    

    weather = weather_interp


    # init_indoor = np.array([23.7, 1291.82276, 1907.9267])
    def run_simulation():
        print(env.N)
        env.reset(seed=env_seed)
        env.weather_data = weather
        # env.x = init_state_pipeinput(env.weather_data[0], init_indoor)
        env.set_crop_state(cBuf=0, cLeaf=0.7*crop_DM, cStem=0.25*crop_DM, cFruit=0.05*crop_DM, tCanSum=0)

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
        # print(Us[:,0])
        # print(Xs[:,0])
        time_end = time.time()
        return time_end - time_start, Xs

    # Time the execution of the function
    elapsed_times = []
    for i in range(1):
        elapsed_time, Xs = run_simulation()
        elapsed_times.append(elapsed_time)
    # save elapsed times to csv
    state_columns = ["co2Air", "co2Top", "tAir", "tTop", "tCan", "tCovIn", "tCovE",
                      "tThScr", "tFlr", "tPipe", "tSo1", "tSo2", "tSo3", "tSo4", "tSo5", 
                      "vpAir", "vpTop", "tLamp", "tIntLamp", "tGroPipe", "tBlScr", "tCan24",
                      "cBuf", "cLeaf", "cStem", "cFruit", "tCanSum", "time"]
    print(np.array(Xs).shape)
    states = pd.DataFrame(np.array(Xs), columns=state_columns)
    states.to_csv("data/bleiswijk/states_pipeinput.csv", index=False)

    weather_cols = ["glob_rad", "temp", "vpout", "co2out", "wind", "tsky", "tso", "dli", "isday", "isday_smooth"]   
    controls_cols = ["uBoil", "uCO2", "uThScr", "uVent", "uLamp", "uBlScr"]
    weather_data = pd.DataFrame(env.weather_data, columns=weather_cols)
    weather_data.to_csv("data/bleiswijk/weather_pipeinput.csv", index=False)
    controls_data = pd.DataFrame(controls, columns=controls_cols)
    controls_data.to_csv("data/bleiswijk/controls_pipeinput.csv", index=False)