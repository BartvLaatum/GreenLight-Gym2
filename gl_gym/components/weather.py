from __future__ import annotations

from pathlib import Path
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Callable
from gl_gym.core.types import WeatherScenario
from gl_gym.environments.utils import load_weather_data

class WeatherRepository:
    def __init__(
        self,
        weather_data_dir: str | Path,
        load_weather_data_fn: Callable[Any, np.ndarray],
    ) -> None:
        self.weather_data_dir = Path(weather_data_dir)
        self.load_weather_data_fn = load_weather_data_fn
        self._cache: dict[tuple, np.ndarray] = {}

    def load(
        self,
        *,
        location: str,
        growth_year: int,
        start_day: int,
        season_length: int,
        pred_horizon: int,
        dt: int,
        nd: int,
    ) -> np.ndarray:
        key = (location, growth_year, start_day)
        if key not in self._cache:
            arr = self.load_weather_data_fn(
                weather_data_dir=self.weather_data_dir,
                location=location,
                growth_year=growth_year,
                start_day=start_day,
                n_days=season_length,
                pred_horizon=pred_horizon,
                h=dt,
                nd=nd,
            )
            self._cache[key] = np.asarray(arr, dtype=np.float64)
        return self._cache[key]


class BaseWeatherSampler(ABC):
    @abstractmethod
    def sample(self, rng: np.random.Generator, options: Dict[str, Any] | None = None) -> WeatherScenario:
        ...

class FixedWeatherSampler(BaseWeatherSampler):
    def __init__(
        self,
        location: str,
        growth_year: int,
        start_day: int,
    ):
        self.location = location
        self.growth_year = growth_year
        self.start_day = start_day

    def sample(self, rng: np.random.Generator, options: Dict[str, Any] | None = None) -> WeatherScenario:
        return WeatherScenario(location=self.location, growth_year=self.growth_year, start_day=self.start_day)

class RandomWeatherSampler(BaseWeatherSampler):
    def __init__(
        self,
        locations: List[str],
        growth_years: List[int],
        start_days: List[int] | range,
    ):
        self.locations = locations
        self.growth_years = growth_years
        self.start_days = start_days

    def sample(self, rng: np.random.Generator, options: Dict[str, Any] | None = None) -> WeatherScenario:
        location = rng.choice(self.locations)
        growth_year = rng.choice(self.growth_years)
        start_day = rng.choice(self.start_days)
        return WeatherScenario(location=location, growth_year=growth_year, start_day=start_day)

class CyclingWeatherSampler(BaseWeatherSampler):
    """
    Deterministic cycling through a predefined list of scenarios.
    Useful for repeated comparable evaluation.
    """
    def __init__(self, scenarios: List[Dict[str, Any]]):
        self.scenarios = [
            WeatherScenario(location=scenario["location"], growth_year=scenario["growth_year"], start_day=scenario["start_day"]) for scenario in scenarios
        ]
        self._i = 0

    def sample(self, rng: np.random.Generator, options: Dict[str, Any] | None = None) -> WeatherScenario:
        if options is not None and "scenario_index" in options:
            idx = int(options["scenario_index"]) % len(self.scenarios)
            return self.scenarios[idx]

        scenario = self.scenarios[self._i % len(self.scenarios)]
        self._i += 1
        return scenario

WEATHER_SAMPLERS = {
    "fixed": FixedWeatherSampler,
    "random": RandomWeatherSampler,
    "cycling": CyclingWeatherSampler,
}
