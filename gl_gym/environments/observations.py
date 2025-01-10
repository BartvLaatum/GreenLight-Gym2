from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
from gymnasium import spaces

from gl_gym.environments.models.greenlight_model import GreenLight

class BaseObservations(ABC):
    """
    Observer class, which gives control over the observations (aka inputs) for our RL agents.
    One can construct observations from GreenLight or from the current and future weather data.
    The model observations are computed in GreenLight,
    the weather observations are extracted from the weather data array.
    """
    def __init__(self,
                ) -> None:
        self.n_obs = None
        self.low = None
        self.high = None
        self.obs_names = None

    @abstractmethod
    def observation_space(self) -> spaces.Box:
        pass

    @abstractmethod
    def compute_obs(self, gl_model: GreenLight) -> np.ndarray:
        """
        Compute, and retrieve observations from GreenLight and the weather.
        """
        pass

class StateObservations(BaseObservations):
    """
    Observer class, which give us control over the observations we want to our RL algorithm to use.
    One can construct observations from GreenLight or from the current and future weather data.
    The model observations are computed in GreenLight,
    the weather observations are extracted from the weather data array.
    """
    def __init__(self) -> None:
        self.model_obs_names = ["co2_air", "co2_top", "temp_air", "temp_top", "can_temp", "covin_temp", "covex_temp",
                                "thScr_temp", "flr_temp", "pipe_temp", "soil1_temp", "soil2_temp", "soil3_temp", "soil4_temp", "soil5_temp", 
                                "vp_air", "vp_top", "lamp_temp", "intlamp_temp", "grow_pipe_temp", "blscr_temp", "24_can_temp",
                                "cBuf", "cleaves", "cstem", "cFruit", "tsum"]
        self.n_obs = len(self.model_obs_names)

    def observation_space(self) -> spaces.Box:
        return spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_obs,), dtype=np.float32)

    def compute_obs(self) -> np.ndarray:
        """
        Compute, and retrieve observations from GreenLight and the weather.
        """
        # all states except the last one which corresponds to time.
        return np.random.rand(self.n_obs)

class WeatherObservations(BaseObservations):
    """
    Observer class, which give us control over the observations we want to our RL algorithm to use.
    One can construct observations from GreenLight or from the current and future weather data.
    The model observations are computed in GreenLight,
    the weather observations are extracted from the weather data array.
    """
    def __init__(self,
                obs_names: List[str],    # observations from the weather
                ) -> None:
        self.obs_names = obs_names
        self.weather_cols = self.weather_vars2idx()
        self.n_obs = len(obs_names)

    def weather_vars2idx(self) -> np.ndarray:
        """
        Functions that converts weather variable names to column indices.
        """
        weather_idx = {"glob_rad": 0, "temp_out": 1, "rh_out": 2, "co2_out": 3, "wind_speed": 4,
                       "temp_sky": 5, "soil_temp": 6, "dli": 7, "is_day": 8, "is_day_smooth": 9}
        return np.array([weather_idx[weather_var] for weather_var in self.obs_names])

    def observation_space(self) -> spaces.Box:
        return spaces.Box(low=-1e-4, high=1e4, shape=(self.n_obs,), dtype=np.float32)

    def compute_obs(self,
                    gl_model: GreenLight,
                    ) -> np.ndarray:
        """
        Compute, and retrieve observations from GreenLight and the weather.
        """
        weather = np.array(gl_model.get_weather())
        weather_obs = weather[self.weather_cols].flatten()
        return weather_obs

class MaskStateObservations(BaseObservations):
    """
    Observer class, which give
    """
    def __init__(self,
                 obs_names: List[str],
                 mask: np.ndarray,
                ) -> None:
        self.obs_names = obs_names
        self.n_obs = len(obs_names)
        self.mask = mask

    def observation_space(self) -> spaces.Box:
        return spaces.Box(low=-1e-4, high=1e4, shape=(self.n_obs,), dtype=np.float32)
    
    def compute_obs(self,
                    gl_model: GreenLight,
                    ) -> np.ndarray:
        """
        Compute, and retrieve observations from GreenLight and the weather.
        """
        state = gl_model.get_state()
        obs = np.delete(state, self.mask)
        return obs[:-1]

class ControlObservations(BaseObservations):
    """Observer module, which gives control over the observations we want to our RL algorithm to use.
    """
    def __init__(self,
                 obs_names: List[str],
                ) -> None:
        self.obs_names = obs_names
        self.n_obs = len(obs_names)

    def observation_space(self):
        return spaces.Box(low=-1e-4, high=1e4, shape=(self.n_obs,), dtype=np.float32)

    def compute_obs(self,
                    gl_model: GreenLight,
                    ) -> np.ndarray:
        """
        Compute, and retrieve observations from GreenLight.
        """
        return np.array(gl_model.get_controls())[[0, 1, 3, 7]]

class ActionObservations(BaseObservations):
    """Observer module, which gives control over the observations we want to our RL algorithm to use.
    
    """
    def __init__(self,
                 obs_names: List[str],
                ) -> None:
        self.obs_names = obs_names
        self.n_obs = len(obs_names)

    def observation_space(self):
        return spaces.Box(low=-1e-4, high=1e4, shape=(self.n_obs,), dtype=np.float32)

    def compute_obs(self,
                    gl_model: GreenLight,
                    ) -> np.ndarray:
        """
        Compute, and retrieve observations from GreenLight and the weather.
        """
        return gl_model.get_setpoints()

class IndoorClimateObservations(BaseObservations):
    """Observer module, which gives control over the observations we want to our RL algorithm to use.
    """
    def __init__(self,
                 obs_names: List[str],
                ) -> None:
        self.obs_names = obs_names
        self.n_obs = len(obs_names)

    def observation_space(self):
        return spaces.Box(low=-1e-4, high=1e4, shape=(self.n_obs,), dtype=np.float32)

    def compute_obs(self,
                    gl_model: GreenLight,
                    ) -> np.ndarray:
        """
        Compute, and retrieve observations from GreenLight and the weather.
        """
        state = np.array(gl_model.get_state())
        return state[[0, 2, 9, 15]]

class BasicCropObservations(BaseObservations):
    """Observer module, which gives control over the observations we want to our RL algorithm to use.
    """
    def __init__(self,
                 obs_names: List[str],
                ) -> None:
        self.obs_names = obs_names
        self.n_obs = len(obs_names)

    def observation_space(self):
        return spaces.Box(low=-1e-4, high=1e4, shape=(self.n_obs,), dtype=np.float32)

    def compute_obs(self,
                    gl_model: GreenLight,
                    ) -> np.ndarray:
        """
        Compute, and retrieve observations from GreenLight and the weather.
        """
        state = np.array(gl_model.get_state())
        return state[[21, 25, 26]]

class WeatherForecastObservations(BaseObservations):
    """Observer module, which gives control over the observations we want to our RL algorithm to use.
    """
    def __init__(
        self,
        obs_names: List[str],
        Np: int,
    ) -> None:
        self.obs_names = obs_names
        self.Np = Np
        self.n_obs = len(obs_names)*Np

    def observation_space(self):
        return spaces.Box(low=-1e-4, high=1e4, shape=(self.n_obs,), dtype=np.float32)

    def compute_obs(self,
                    gl_model: GreenLight,
                    ) -> np.ndarray:
        """
        Compute, and retrieve observations from GreenLight and the weather.
        """
        forecast = []
        for i in range(1, self.Np+1):
            forecast += gl_model.get_weather_pred(i)[:5]        # Only the first 5 weather variables
        return np.array(forecast)

