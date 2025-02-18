import argparse
from gl_gym.environments.tomato_env import TomatoEnv
from gl_gym.common.utils import load_env_params
import numpy as np
import pandas as pd
from time import time

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


def init_mat_state(d0, indoor, time_in_days=0):
    # Initialize the state as a NumPy array of zeros with the appropriate size
    state = np.zeros(28)  # Assuming 28 elements based on the original function

    state[0] = indoor[1]        # co2Air
    state[1] = state[0]     # co2Top
    state[2] = indoor[0]         # tAir
    state[3] = state[2]     # tTop
    state[4] = state[2] + 4 # tCan
    state[5] = state[2]     # tCovIn
    state[6] = state[2]     # tCovE
    state[7] = state[2]     # tThScr
    state[8] = state[2]     # tFlr
    state[9] = d0[10]     # tPipe
    state[10] = state[2]    # tSoil1
    state[11] = 0.25 * (3. * state[2] + d0[6])      # tSoil2
    state[12] = 0.25 * (2. * state[2] + 2 * d0[6])  # tSoil3
    state[13] = 0.25 * (state[2] + 3 * d0[6])       # tSoil4
    state[14] = d0[6]                               # tSoil5
    state[15] = indoor[2]   # vpAir
    state[16] = state[15]   # vpTop
    state[17] = state[2]    # tLamp
    state[18] = state[2]    # tIntLamp
    state[19] = state[2]    # tGroPipe
    state[20] = state[2]    # tBlScr
    state[21] = state[4]    # tCan24
    state[22] = 0.          # cBuf
    state[23] = 9.5283e4    # cLeaf
    state[24] = 2.5107e5    # cStem
    state[25] = 5.5338e4    # cFruit
    state[26] = 3.0978e3    # tCanSum
    state[27] = time_in_days    # time

    return state

def set_matlab_params(params):
    params[79] = 0.6                # tauThScrNir
    params[108] = 44.*params[46]    # pBoil
    params[109] = 720.              # phiExtCO2
    params[165] = 0.88              # epsGroPipe
    params[170] = 44. * params[46]  # pBoilGro
    params[145] = 300_000           # cFruitMax
    return params

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="TomatoEnv")
    parser.add_argument("--env_config_path", type=str, default="gl_gym/configs/envs/")
    parser.add_argument("--save_state", default=True, action=argparse.BooleanOptionalAction, help="Save the state results")
    parser.add_argument("--save_time", default=True, action=argparse.BooleanOptionalAction, help="Save the timing")
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
    env_base_params["nu"] = 6
    env_base_params["nd"] = 14
    env_base_params["dt"] = 300
    env_base_params["pred_horizon"] = 0

    dt_data = 300
    Ns_data = int(86400*env_base_params['season_length']/dt_data) + 1

    # Initialize the environment with both parameter dictionaries
    env = TomatoEnv(base_env_params=env_base_params, **env_specific_params)

    controls = pd.read_csv('data/AgriControl/comparison/matlab/controls_pipe2009.csv').values[:Ns_data, :6]
    weather = pd.read_csv('data/AgriControl/comparison/matlab/weather_pipe2009.csv').values[:Ns_data, :]
    # weather = interpolate_weather_data(weather, env_base_params)
    indoor = [23.7, 1291.822759273841, 1907.926700562060]

    def run_simulation():
        env.reset(seed=env_seed)
        env.weather_data = weather
        env.p = set_matlab_params(env.p)
        env.x = init_mat_state(env.weather_data[0], indoor)
        # env.set_crop_state(cBuf=0, cLeaf=0.7*crop_DM, cStem=0.25*crop_DM, cFruit=0.05*crop_DM, tCanSum=0)

        done = False
        Xs = [np.copy(env.x)]
        Us = [np.copy(env.u)]
        start = time()
        for _ in range(env.N):

            # x, done = env.step_raw_control(controls[env.timestep])
            x, done = env.step_raw_control_pipeinput(controls[env.timestep])
            Xs.append(x)
            Us.append(env.u)
        Xs = np.array(Xs)
        Us = np.array(Us)
        end = time()
        elapsed_time = (end - start)
        return Xs, Us, elapsed_time

    # Time the execution of the function
    elapsed_times = []
    for _ in range(10):
        Xs, Us, elapsed_time = run_simulation()
        elapsed_times.append(elapsed_time)
    df = pd.DataFrame(elapsed_times, columns=["elapsed_time"])
    df.to_csv("data/AgriControl/run_times/gl_gym.csv", index=False)


    if args.save_state:

        # save elapsed times to csv
        state_columns = [
            "co2Air", "co2Top", "tAir", "tTop", "tCan", "tCovIn", "tCovE",
            "tThScr", "tFlr", "tPipe", "tSo1", "tSo2", "tSo3", "tSo4", "tSo5", 
            "vpAir", "vpTop", "tLamp", "tIntLamp", "tGroPipe", "tBlScr", "tCan24",
            "cBuf", "cLeaf", "cStem", "cFruit", "tCanSum", "time"
        ]

        states = pd.DataFrame(np.array(Xs), columns=state_columns)
        states.to_csv("data/AgriControl/comparison/gl_gym/states_pipe2009.csv", index=False)

        weather_cols = [
            "glob_rad", "temp", "vpout", "co2out", 
            "wind", "tsky", "tso", "dli", 
            "isday", "isday_smooth", "tPipe", 
            "tGroPipe", "pipeSwitchOff", "groPipeSwitchOff"
        ]

        weather_data = pd.DataFrame(env.weather_data, columns=weather_cols)
        weather_data.to_csv("data/AgriControl/comparison/gl_gym/weather_pipe2009.csv", index=False)

        controls_cols = ["uBoil", "uCO2", "uThScr", "uVent", "uLamp", "uBlScr"]
        controls_data = pd.DataFrame(controls, columns=controls_cols)
        controls_data.to_csv("data/AgriControl/comparison/gl_gym/controls_pipe2009.csv", index=False)
