
from typing import Any, Dict, List, Optional, Tuple, SupportsFloat

import numpy as np
from gymnasium import spaces

from gl_gym.environments.base_env import GreenLightEnv
from gl_gym.environments.observations import *
from gl_gym.environments.rewards import BaseReward, EconomicReward
from gl_gym.environments.models.greenlight_model import GreenLight
from gl_gym.environments.utils import load_weather_data, init_state
from gl_gym.environments.parameters import init_default_params

REWARDS = {"EconomicReward": EconomicReward}

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

class TomatoEnv(GreenLightEnv):
    def __init__(self,
        reward_function: str,                           # reward function
        observation_module: str,                   # observation function
        eval_options: Dict[str, Any],                   # days for evaluation
        reward_params: Dict[str, Any] = {},             # reward function arguments
        base_env_params: Dict[str, Any] = {},
        ) -> None:
        super(TomatoEnv, self).__init__(**base_env_params)
        self.np = 208


        # set year and days for the evaluation environment 
        self.eval_options = eval_options

        # initialise the observation and action spaces
        self.observation_module = self._init_observations(observation_module)
        self.observation_space = self._generate_observation_space()
        self.action_space = self._generate_action_space()

        self.gl_model = GreenLight(self.nx, self.nu, self.nd, self.np, self.dt)

        # initialise the reward function
        self.reward = self._init_rewards(reward_function, reward_params)


    def _terminalState(self) -> bool:
        """
        Function that checks whether the simulation has reached a terminal state.
        Terminal states are reached when the simulation has reached the end of the growing season.
        """
        if self.timestep >= self.N:
            return True
        return False

    def _init_observations(
        self,
        observation_module: str,
    ) -> List[BaseObservations]:
        return OBSERVATION_MODULES[observation_module]()

    def _generate_observation_space(self) -> spaces.Box:
        return self.observation_module.observation_space()

    def _generate_action_space(self) -> spaces.Box:
        return spaces.Box(low=-1, high=1, shape=(self.nu,), dtype=np.float32)

    def _init_rewards(self, reward_function: str, reward_params: Dict[str, Any]) -> BaseReward:
        return REWARDS[reward_function](**reward_params)

    def _get_reward(self) -> SupportsFloat:
        # TODO: implement reward function
        return 0
        # return self.reward.compute_reward(self.gl_model)

    def _scale(self, action, action_min, action_max):
        return (action + 1) * (action_max - action_min) / 2 + action_min

    def action_to_control(self, action: np.ndarray) -> np.ndarray:
        """
        Function that converts the action to control inputs.
        """
        return np.clip(self.u + action*self.delta_u_max, self.u_min, self.u_max) 


    def step(self, action: np.ndarray) -> Tuple[np.ndarray, SupportsFloat, bool, bool, Dict[str, Any]]:
        # scale the action from controller (between -1, 1) to (u_min, u_max)
        self.u = self.action_to_control(action)

        # self.weather_data
        self.x =self.gl_model.evalF(self.x, self.u, self.weather_data[self.timestep], self.p)

        obs = self._get_obs()
        if self._terminalState():
            self.terminated = True
        # compute reward
        reward = self._get_reward()

        # additional information to return
        info = self._get_info()
        self.timestep += 1
        return (
                obs,
                reward, 
                self.terminated, 
                False,
                info
                )

    def step_raw_control(self, control: np.ndarray):
        # scale the action from controller (between -1, 1) to (u_min, u_max)
        self.u = control

        # self.weather_data
        self.x =self.gl_model.evalF(self.x, self.u, self.weather_data[self.timestep], self.p)

        # obs = self._get_obs()
        if self._terminalState():
            self.terminated = True
        # compute reward
        # reward = self._get_reward()

        # additional information to return
        # info = self._get_info()
        self.timestep += 1
        return (
                self.x,
                self.terminated, 
                )


    def _get_obs(self):
        return self.observation_module.compute_obs()

    def _get_info(self) -> Dict[str, Any]:
        # state = self.gl_model.get_state()
        return {
            "Profit": self.reward.profit,
            "Gains": self.reward.gains,
            "Variable costs": self.reward.variable_costs,
            "Fixed costs": self.reward.fixed_costs,
            "Control inputs": self.u,
        }

    def set_crop_state(self, cBuf: float, cLeaf: float, cStem: float, cFruit: float, tCanSum: float):
        self.x[22] = cBuf
        self.x[23] = cLeaf
        self.x[24] = cStem
        self.x[25] = cFruit
        self.x[26] = tCanSum

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


        # load in weather data for specific simulation
        self.weather_data = load_weather_data(
            self.weather_data_dir,
            self.location,
            self.data_source,
            self.growth_year,
            self.start_day,
            self.season_length,
            self.pred_horizon+1,
            self.dt,
            self.nd
        )
        self.u = np.zeros(self.nu)
        self.x = init_state(self.weather_data[0])

        self.timestep = 0
        self.p = init_default_params(self.np)
        # compute days since 01-01-0001
        # as time indicator by the model
        timeInDays = self._get_time_in_days()


        self.terminated = False
        return self._get_obs(), {}

