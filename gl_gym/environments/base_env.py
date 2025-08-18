from datetime import date
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding

from gl_gym.environments.observations import BaseObservations
from gl_gym.environments.rewards import BaseReward

class GreenLightEnv(gym.Env):
    """
    Python base class that functions as a wrapper environment for bindings between python and C++. 

    Args:
        weather_data_dir (str): path to weather data
        location (str): location of the recorded weather data
        nx (int): number of states 
        nd (int): number of disturbances
        h (float): [s] time step for the GreenLight solver
        time_interval (int): [s] time interval in between observations
        pred_horizon (int): [days] number of future weather predictions
        n_setpoints (int): number of setpoints we control
        season_length (int): end day of the simulation. Defaults to 350.
        start_train_year (int, optional): start year for training. Defaults to 2023.
        end_train_year (int, optional): end year for training. Defaults to 2023.
        start_train_day (int, optional): start day for training. Defaults to 265.
        end_train_day (int, optional): end day for training. Defaults to 284.
        reward_function (str, optional): reward function to use. Defaults to "None".
        training (bool, optional): whether we are training or testing. Defaults to True.
        options (Dict[str, Any], optional): options for the GreenLight model. Defaults to {}.
    """
    # gl_model: GreenLight
    growth_year: int
    start_day: int
    reward_function: BaseReward
    def __init__(
        self,
        weather_data_dir: str,          # path to weather data
        location: str,                  # location of the recorded weather data
        num_params: int,                # number of model parameters
        nx: int,                        # number of states
        nu: int,                        # number of control inputs
        nd: int,                        # number of disturbances
        dt: float,                      # [s] time step for the underlying GreenLight solver
        u_min: List[float],             # min bound for the control inputs
        u_max: List[float],             # max for the control inputs
        delta_u_max: float,             # max change for the control inputs
        pred_horizon: int,              # [days] number of future weather predictions
        season_length: int = 60,        # season length
        start_train_year: int = 2023,   # start year for training
        end_train_year: int = 2023,     # end year for training
        start_train_day: int = 265, 	# start day of the year for training
        end_train_day: int = 284,       # end day of the year for training
        training: bool = True,          # whether we are training or testing
    ) -> None:
        super(GreenLightEnv, self).__init__()

        # number of seconds in the day
        self.c = 86400

        # arguments that are kept the same over various simulations
        self.num_params = num_params
        self.nx = nx
        self.nu = nu
        self.nd = nd
        self.u_min = np.array(u_min, dtype=np.float32)
        self.u_max = np.array(u_max, dtype=np.float32)
        self.delta_u_max = np.ones(self.nu, dtype=np.float32) * delta_u_max
        self.weather_data_dir = weather_data_dir
        self.location = location
        self.dt = dt
        self.pred_horizon = pred_horizon
        self.Np = int(self.pred_horizon * self.c/self.dt)
        self.train_years = list(range(start_train_year, end_train_year+1))
        self.train_days = list(range(start_train_day, end_train_day+1))

        self.training = training
        self.eval_idx = 0

        self.season_length = season_length
        self.N = int(self.season_length * self.c/self.dt)

        # lower and upper bounds for air temperature, co2 concentration, humidity
        self.obs_low = None
        self.obs_high = None

    @abstractmethod
    def step(self, action: np.ndarray
                ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Input: action
        Output: observation, reward, done, truncated, info
        """
        pass

    @abstractmethod
    def _get_info(self) -> Dict[str, Any]:
        """	
        Get information about the current state of the environment.
        """
        pass

    @abstractmethod
    def _init_rewards(self) -> BaseReward:
        """
        Initialize the rewards for the environment.
        """
        pass

    @abstractmethod
    def _init_observations(
                            self,
                            model_obs_vars: Optional[List[str]] = None,
                            weather_obs_vars: Optional[List[str]] = None,
                            Np: Optional[int] = None
                            ) -> List[BaseObservations]:
        
        pass

    @abstractmethod
    def _generate_observation_space(self) -> spaces.Dict:
        pass

    @abstractmethod
    def _get_obs(self):
        pass

    @abstractmethod
    def _terminalState(self) -> bool:
        pass

    def _get_time(self):
        """
        Get the current time in seconds.
        """
        return self.gl_model.get_time()

    def _get_time_in_days(self) -> float:
        """
        Get time in days since 01-01-0001 upto the starting day of the simulation.
        """
        d0 = date(1, 1, 1)
        d1 = date(self.growth_year, 1, 1)
        delta = d1 - d0
        return delta.days + self.start_day

    def _scale(self, action, action_min, action_max):
        """
        Min-max scaler [0,1]. Used for the action space.
        """
        return (action-action_min)/(action_max-action_min)

    def _reset_eval_idx(self):
        self.eval_idx = 0

    def increase_eval_idx(self):
        self.eval_idx += 1

    def set_seed(self, seed):
        """
        Seed the environment.
        """
        self._np_random, self._np_random_seed = seeding.np_random(seed)

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Container function that resets the environment.
        Calls super().reset(seed=seed) to reset the environment.
        To seed the Gymnasium environment.
        """
        super().reset(seed=seed)
        return np.array([]), {}