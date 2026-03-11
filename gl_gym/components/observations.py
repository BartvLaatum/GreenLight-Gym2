from abc import ABC, abstractmethod
from typing import List

import numpy as np
from gymnasium import spaces

from gl_gym.environments.utils import co2dens2ppm, vaporPres2rh
from gl_gym.core.types import StepContext

class BaseObservations(ABC):
    def __init__(self, env) -> None:
        self.Np = env.Np
        self.nu = env.nu

    @property
    def key(self) -> str:
        ...

    @property
    def space(self) -> spaces.Box:
        ...

    @abstractmethod
    def compute_obs(self, ctx: StepContext):
        ...

class StateObservations(BaseObservations):
    """
    Observer class, which give us control over the observations we want to our RL algorithm to use.
    One can construct observations from GreenLight or from the current and future weather data.
    The model observations are computed in GreenLight,
    the weather observations are extracted from the weather data array.
    """
    @property
    def key(self) -> str:
        return "StateObservations"

    @property
    def space(self) -> spaces.Box:
        return spaces.Box(low=-1e-4, high=1e4, shape=(27,), dtype=np.float32)

    def compute_obs(self, ctx: StepContext) -> np.ndarray:
        return np.array(ctx.x)

class IndoorClimateObservations(BaseObservations):
    """Observer module, which gives control over the observations we want to our RL algorithm to use.
    """
    @property
    def key(self) -> str:
        return "IndoorClimateObservations"

    @property
    def space(self) -> spaces.Box:
        return spaces.Box(low=-1e-4, high=1e4, shape=(4,), dtype=np.float32)

    @property
    def obs_names(self) -> List[str]:
        return ["co2_air", "temp_air", "rh_air", "pipe_temp"]

    def compute_obs(self, ctx: StepContext) -> np.ndarray:
        """
        Compute, and retrieve observations from GreenLight and the weather.
        """
        climate_obs = np.array(ctx.x)[[0, 2, 15, 9]]
        climate_obs[0] = co2dens2ppm(climate_obs[1], climate_obs[0]*1e-6)
        climate_obs[2] = vaporPres2rh(climate_obs[1], climate_obs[2])
        return climate_obs

class BasicCropObservations(BaseObservations):
    """Observer module, which gives control over the observations we want to our RL algorithm to use.
    """
    @property
    def key(self) -> str:
        return "BasicCropObservations"

    @property
    def space(self) -> spaces.Box:
        return spaces.Box(low=-1e-4, high=1e4, shape=(3,), dtype=np.float32)

    @property
    def obs_names(self) -> List[str]:
        return ["tCan24", "cFruit", "tCanSum"]

    def compute_obs(self, ctx: StepContext) -> np.ndarray:
        """
        Compute, and retrieve observations from GreenLight and the weather.
        """
        crop_obs = np.array(ctx.x)[[21, 25, 26]]
        return crop_obs

class ControlObservations(BaseObservations):
    """Observer module, which gives control over the observations we want to our RL algorithm to use.
    """
    @property
    def key(self) -> str:
        return "ControlObservations"

    @property
    def space(self) -> spaces.Box:
        return spaces.Box(low=0., high=1., shape=(self.nu,), dtype=np.float32)

    @property
    def obs_names(self) -> List[str]:
        return ["uBoil", "uCO2", "uThScr", "uVent", "uLamp", "uBlScr"]

    def compute_obs(self, ctx: StepContext) -> np.ndarray:
        """
        Compute, and retrieve observations from GreenLight.
        """
        return ctx.u

class WeatherObservations(BaseObservations):
    """
    Observer class, which give us control over the observations we want to our RL algorithm to use.
    One can construct observations from GreenLight or from the current and future weather data.
    The model observations are computed in GreenLight,
    the weather observations are extracted from the weather data array.
    """
    @property
    def key(self) -> str:
        return "WeatherObservations"

    @property
    def space(self) -> spaces.Box:
        return spaces.Box(low=-1e-4, high=1e4, shape=(5,), dtype=np.float32)

    @property
    def obs_names(self) -> List[str]:
        return ["radiation", "temperature", "rh", "co2", "wind"]

    def compute_obs(self, ctx: StepContext) -> np.ndarray:
        """
        Compute, and retrieve observations from GreenLight and the weather.
        """
        weather_obs = np.copy(ctx.d[ctx.t, :5])
        weather_obs[2] = vaporPres2rh(weather_obs[1], weather_obs[2])
        weather_obs[3] = co2dens2ppm(weather_obs[1], weather_obs[3]*1e-6)
        return weather_obs

class TimeObservations(BaseObservations):
    """Observer module, which gives control over the observations we want to our RL algorithm to use.
    """
    name = "TimeObservations"
    @property
    def key(self) -> str:
        return "TimeObservations"

    @property
    def space(self) -> spaces.Box:
        return spaces.Box(low=-1e-4, high=1e4, shape=(4,), dtype=np.float32)

    @property
    def obs_names(self) -> List[str]:
        return ["day_of_year_sin", "day_of_year_cos", "hour_of_day_sin", "hour_of_day_cos"]

    def compute_obs(self, ctx: StepContext) -> np.ndarray:
        """
        Computes time-based observations. Both day of year and hour of day are normalized to the range [0, 1], using sin and cos.
        Such that the model can learn the periodicity of the day and year.
        The timestep indicates the current time step.
        """
        day_of_year_sin = np.sin(2 * np.pi * ctx.day_of_year / 365.0)
        day_of_year_cos = np.cos(2 * np.pi * ctx.day_of_year / 365.0)

        hour_of_day_sin = np.sin(2 * np.pi * ctx.hour_of_day / 24.0)
        hour_of_day_cos = np.cos(2 * np.pi * ctx.hour_of_day / 24.0)

        return np.array([day_of_year_sin, day_of_year_cos, hour_of_day_sin, hour_of_day_cos])

class WeatherForecastObservations(BaseObservations):
    """Observer module, which gives control over the observations we want to our RL algorithm to use.
    """
    @property
    def key(self) -> str:
        return "WeatherForecastObservations"

    @property
    def space(self) -> spaces.Box:
        return spaces.Box(low=-1e-4, high=1e4, shape=(5*self.Np,), dtype=np.float32)

    @property
    def obs_names(self) -> List[str]:
        return ["radiation", "temperature", "rh", "co2", "wind"] * self.Np

    def compute_obs(self, ctx: StepContext) -> np.ndarray:
        """
        Compute, and retrieve observations from GreenLight and the weather.
        """
        forecast = []
        for i in range(1, self.Np+1):
            forecast.extend(ctx.d[ctx.t+i][0:5])        # Only the first 5 weather variables
        return np.array(forecast)      # user-defined via import path


OBSERVATION_MODULES = {
    "StateObservations": StateObservations,
    "IndoorClimateObservations": IndoorClimateObservations,
    "BasicCropObservations": BasicCropObservations,
    "ControlObservations": ControlObservations,
    "WeatherObservations": WeatherObservations,
    "WeatherForecastObservations": WeatherForecastObservations,
    "TimeObservations": TimeObservations,
}
