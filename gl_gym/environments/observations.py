from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
from gymnasium import spaces

from gl_gym.environments.utils import co2dens2ppm, vaporPres2rh

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
    def compute_obs(self) -> np.ndarray:
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
        self.obs_names = ["co2_air", "co2_top", "temp_air", "temp_top", "can_temp", "covin_temp", "covex_temp",
                                "thScr_temp", "flr_temp", "pipe_temp", "soil1_temp", "soil2_temp", "soil3_temp", "soil4_temp", "soil5_temp", 
                                "vp_air", "vp_top", "lamp_temp", "intlamp_temp", "grow_pipe_temp", "blscr_temp", "24_can_temp",
                                "cBuf", "cleaves", "cstem", "cFruit", "tsum"]
        self.n_obs = len(self.obs_names)

    def observation_space(self) -> spaces.Box:
        return spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_obs,), dtype=np.float32)

    def compute_obs(self) -> np.ndarray:
        """
        Compute, and retrieve observations from GreenLight and the weather.
        """
        # all states except the last one which corresponds to time.
        return np.random.rand(self.n_obs)

class IndoorClimateObservations(BaseObservations):
    """Observer module, which gives control over the observations we want to our RL algorithm to use.
    """
    def __init__(self, env) -> None:
        self.env = env
        self.obs_names = ["co2_air", "temp_air","rh_air",  "pipe_temp"]
        self.n_obs = len(self.obs_names)

    def observation_space(self):
        return spaces.Box(low=-1e-4, high=1e4, shape=(self.n_obs,), dtype=np.float32)

    def compute_obs(self) -> np.ndarray:
        """
        Compute, and retrieve observations from GreenLight and the weather.
        """
        climate_obs = np.array(self.env.x)[[0, 2, 15, 9]]
        climate_obs[0] = co2dens2ppm(climate_obs[1], climate_obs[0]*1e-6)
        climate_obs[2] = vaporPres2rh(climate_obs[1], climate_obs[2])
        return climate_obs

class BasicCropObservations(BaseObservations):
    """Observer module, which gives control over the observations we want to our RL algorithm to use.
    """
    def __init__(self, env) -> None:
        self.env = env
        self.obs_names = ["24CanTemp", "cFruit", "tSum"]
        self.n_obs = len(self.obs_names)

    def observation_space(self):
        return spaces.Box(low=-1e-4, high=1e4, shape=(self.n_obs,), dtype=np.float32)

    def compute_obs(self) -> np.ndarray:
        """
        Compute, and retrieve observations from GreenLight and the weather.
        """
        crop_obs = np.array(self.env.x)[[21, 25, 26]]
        return crop_obs

class ControlObservations(BaseObservations):
    """Observer module, which gives control over the observations we want to our RL algorithm to use.
    """
    def __init__(self, env) -> None:
        self.env = env
        self.obs_names = ["uBoil", "uCo2", "uThScr", "uVent", "uLamp", "uBlScr"]
        self.n_obs = len(self.obs_names)

    def observation_space(self):
        return spaces.Box(low=0., high=1., shape=(self.n_obs,), dtype=np.float32)

    def compute_obs(self) -> np.ndarray:
        """
        Compute, and retrieve observations from GreenLight.
        """
        return self.env.u

class WeatherObservations(BaseObservations):
    """
    Observer class, which give us control over the observations we want to our RL algorithm to use.
    One can construct observations from GreenLight or from the current and future weather data.
    The model observations are computed in GreenLight,
    the weather observations are extracted from the weather data array.
    """
    def __init__(self, env) -> None:
        self.env = env
        self.obs_names = ["glob_rad", "temp_out", "rh_out", "co2_out", "wind_speed"]
        self.n_obs = len(self.obs_names)

    def observation_space(self) -> spaces.Box:
        return spaces.Box(low=-1e-4, high=1e4, shape=(self.n_obs,), dtype=np.float32)

    def compute_obs(self) -> np.ndarray:
        """
        Compute, and retrieve observations from GreenLight and the weather.
        """
        weather_obs = np.copy(self.env.weather_data[self.env.timestep][0:5])
        weather_obs[2] = vaporPres2rh(weather_obs[1], weather_obs[2])
        weather_obs[3] = co2dens2ppm(weather_obs[1], weather_obs[3]*1e-6)
        return weather_obs

class TimeObservations(BaseObservations):
    """Observer module, which gives control over the observations we want to our RL algorithm to use.
    """
    def __init__(self, env) -> None:
        self.env = env
        self.obs_names = ["timestep", "day of year sin", "day of year cos", "hour of day sin", "hour of day cos"]
        self.n_obs = len(self.obs_names)

    def observation_space(self):
        return spaces.Box(low=-1e-4, high=1e4, shape=(self.n_obs,), dtype=np.float32)

    def compute_obs(self) -> np.ndarray:
        """
        Computes time-based observations. Both day of year and hour of day are normalized to the range [0, 1], using sin and cos.
        Such that the model can learn the periodicity of the day and year.
        The timestep indicates the current time step.
        """
        day_of_year_sin = np.sin(2 * np.pi * self.env.day_of_year / 365.0)
        day_of_year_cos = np.cos(2 * np.pi * self.env.day_of_year / 365.0)

        hour_of_day_sin = np.sin(2 * np.pi * self.env.hour_of_day / 24.0)
        hour_of_day_cos = np.cos(2 * np.pi * self.env.hour_of_day / 24.0)

        return np.array([self.env.timestep, day_of_year_sin, day_of_year_cos, hour_of_day_sin, hour_of_day_cos])

class WeatherForecastObservations(BaseObservations):
    """Observer module, which gives control over the observations we want to our RL algorithm to use.
    """
    def __init__(self, env) -> None:
        self.env = env
        self.n_obs = 5*self.env.Np
        self.obs_names = ["glob_rad", "temp_out", "rh_out", "co2_out", "wind_speed"]*self.env.Np


    def observation_space(self):
        return spaces.Box(low=-1e-4, high=1e4, shape=(self.n_obs,), dtype=np.float32)

    def compute_obs(self) -> np.ndarray:
        """
        Compute, and retrieve observations from GreenLight and the weather.
        """
        forecast = []
        for i in range(1, self.env.Np+1):
            forecast.extend(self.env.weather_data[self.env.timestep+i][0:5])        # Only the first 5 weather variables
        return np.array(forecast)