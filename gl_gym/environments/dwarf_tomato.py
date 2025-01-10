
from typing import Any, Dict, List, Optional, Tuple, SupportsFloat

import numpy as np
from gymnasium import spaces

from gl_gym.environments.base_env import GreenLightEnv
from gl_gym.environments.observations import *
from gl_gym.environments.rewards import BaseReward, EconomicReward
from gl_gym.environments.models.greenlight_model import GreenLight
from gl_gym.environments.utils import load_weather_data

OBSERVATION_MODULES = {
    "StateObservations": StateObservations,
    "WeatherObservations": WeatherObservations,
    "MaskStateObservations": MaskStateObservations,
    "IndoorClimateObservations": IndoorClimateObservations,
    "BasicCropObservations": BasicCropObservations,
    "ActionObservations": ActionObservations,
    "ControlObservations": ControlObservations,
    "WeatherForecastObservations": WeatherForecastObservations,
}

class DwarfTomatoes(GreenLightEnv):
    def __init__(self,
        setpoints_low: List[int],                       # lower bounds for setpoints
        setpoints_high: List[int],                      # upper bounds for setpoints
        reward_function: str,                           # reward function
        selected_obs_mods: List[str],                   # observation function
        eval_options: Dict[str, Any],     # days for evaluation
        model_obs_vars: List[str] | None = None,        # model observation variables
        weather_obs_vars: List[str] | None = None,      # weather observation variables
        eval_years: List[int] | None  = None,            # years for evaluation
        reward_kwargs: Dict[str, Any] = {},             # reward function arguments
        obs_mod_kwargs: Dict[str, Any] = {}, # observation function arguments

        **base_env_kwargs,
        ) -> None:
        super(DwarfTomatoes, self).__init__(**base_env_kwargs)

        self.setpoints_low = np.array(setpoints_low)
        self.setpoints_high = np.array(setpoints_high)
        self.n_setpoints = len(setpoints_low)
        self.setpoints = np.zeros(self.n_setpoints)

        # set year and days for the evaluation environment 
        self.eval_options = eval_options
        # self.eval_years = eval_years
        # self.eval_days = eval_days

        # initialise the observation and action spaces
        self.observation_modules = self._init_observations(selected_obs_mods, obs_mod_kwargs)
        self.observation_space = self._generate_observation_space()
        self.action_space = self._generate_action_space()

        self.gl_model = GreenLight(self.nx, self.solver_steps, self.h, float(self.solver_steps))
        
        self.end_temp_sum = self.gl_model.get_end_temp_sum()

        # initialise the reward function
        self.reward = self._init_rewards(reward_function, reward_kwargs)

    def _terminalState(self) -> bool:
        """
        Function that checks whether the simulation has reached a terminal state.
        Terminal states are reached when the simulation has reached the end of the growing season.
        Or when there are nan or inf in the state values.
        """
        if self.gl_model.timestep >= self.N:
            return True
        # check whether the plant has stopped growing this takes some time... better to do it only once..
        # elif self.gl_model.get_can_temp_sum() > self.end_temp_sum:
        #     return True
        return False

    def _init_observations(
                            self,
                            selected_obs_mods: List[str],
                            obs_mod_kwargs: Dict[str, Any],
                        ) -> List[BaseObservations]:
        return [OBSERVATION_MODULES[module](**obs_mod_kwargs[module]) for module in selected_obs_mods]

    def _generate_observation_space(self) -> spaces.Box:
        spaces_low_list = []
        spaces_high_list = []

        for module in self.observation_modules:
            module_name = module.__class__.__name__.lower()
            module_obs_space = module.observation_space()
            spaces_low_list.append(module_obs_space.low)
            spaces_high_list.append(module_obs_space.high)

        low = np.concatenate(spaces_low_list, axis=0)  # Concatenate low bounds
        high = np.concatenate(spaces_high_list, axis=0)  # Concatenate high bounds

        return spaces.Box(low=low, high=high, dtype=np.float32)

    def _generate_action_space(self) -> spaces.Box:
        return spaces.Box(low=-1, high=1, shape=(self.n_setpoints,), dtype=np.float32)

    def _init_rewards(self, reward_function: str, reward_kwargs: Dict[str, Any]) -> BaseReward:
        reward_types = {"EconomicReward": EconomicReward}
        return reward_types[reward_function](**reward_kwargs, \
                gl_model=self.gl_model,\
                timesteps_in_day=86400//self.time_interval)

    def _get_reward(self) -> SupportsFloat:
        return self.reward.compute_reward(self.gl_model)

    def _scale(self, action, action_min, action_max):
        return (action + 1) * (action_max - action_min) / 2 + action_min

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, SupportsFloat, bool, bool, Dict[str, Any]]:
        # scale the action from controller (between -1, 1) to actual setpoints
        # (perhaps use a max-delta for the actions to get more realistic solutions)
        self.setpoints = self._scale(action, self.setpoints_low, self.setpoints_high)
        self.gl_model.step(self.setpoints)

        obs = self._get_obs()
        if self._terminalState():
            self.terminated = True
        # compute reward
        reward = self._get_reward()

        # if self.gl_model.timestep % 24 == 0:
        #     reward = self._get_reward()
        #     self.gl_model.reset_consumptions()
        #     self.gl_model.reset_dli()
        # else:
        #     reward = self.reward.compute_hourly_reward()

        # additional information to return
        info = self._get_info()

        return (
                obs,
                reward, 
                self.terminated, 
                False,
                info
                )

    def _get_obs(self):
        obs = []
        for module in self.observation_modules:
            obs.append(module.compute_obs(self.gl_model))
        obs = np.concatenate(obs, axis=0)
        # print('shape',obs.shape)
        return obs

    def _get_info(self) -> Dict[str, Any]:
        # state = self.gl_model.get_state()
        return {
            "Profit": self.reward.profit,
            "Gains": self.reward.gains,
            "Variable costs": self.reward.variable_costs,
            "Fixed costs": self.reward.fixed_costs,
            "Control inputs": self.gl_model.get_controls(),
            "Action": self.setpoints,
        }

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        # pick a random growth year and start day if we are training
        if self.training:
            self.growth_year = self._np_random.choice(self.train_years)
            self.start_day = self._np_random.choice(self.train_days)
        else:
            print("Evaluating")
            
            self.growth_year = self._np_random.choice(self.eval_options["eval_years"])
            self.start_day = self._np_random.choice(self.eval_options["eval_days"])
            self.location = self.eval_options["location"]
            self.data_source = self.eval_options["data_source"]
            self.increase_eval_idx()

        # given the start day, compute the season length
        # the end day is fixed to 16th of December (as for the Bleiswijk experiments)
        # self.season_length = self.end_day - self.start_day

        # load in weather data for specific simulation
        self.weatherData = load_weather_data(
                self.weather_data_dir,
                self.location,
                self.data_source,
                self.growth_year,
                self.start_day,
                self.season_length,
                self.pred_horizon,
                self.h,
                self.nd
                )

        # compute days since 01-01-0001
        # as time indicator by the model
        timeInDays = self._get_time_in_days()

        # reset the GreenLight model starting settings
        self.gl_model.reset(timeInDays, self.weatherData)

        self.terminated = False
        return self._get_obs(), {}
