from __future__ import annotations

from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class WeatherScenario:
    location: str       # location of the recorded weather data
    growth_year: int    # growth year (i.e., year of the simulation)
    start_day: int      # start day of the year (i.e., day of the year of the simulation)

@dataclass
class EnvState:
    x: np.ndarray
    u_full: np.ndarray
    p: np.ndarray
    timestep: int
    scenario: WeatherScenario
    weather_data: np.ndarray

@dataclass(frozen=True)
class StepContext:
    t: int # timestep
    dt: int # [s] discretization time (i.e., control interval) for the underlying GreenLight solver
    Np: int # number of future weather predictions
    
    x_prev: np.ndarray          # previous state vector
    x: np.ndarray               # current state vector
    u: np.ndarray               # control input vector from the agent
    p: np.ndarray               # model parameters vector
    d: np.ndarray               # weather disturbances for all timesteps

    hour_of_day: float          # [h] hour of the days at timestep t
    day_of_year: float          # [d] day of the year at timestep t
    
@dataclass(frozen=True)
class RewardContext(StepContext):
    obs: dict[str, np.ndarray]
    constraints_low: np.ndarray | None = None
    constraints_high: np.ndarray | None = None
