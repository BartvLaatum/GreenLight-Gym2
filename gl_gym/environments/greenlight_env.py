from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, Union, SupportsFloat
from collections import OrderedDict
from pathlib import Path

import numpy as np
import casadi as ca
import gymnasium as gym
from gymnasium import spaces, Space

from gl_gym.components.actions import NamedControlActionScheme
from gl_gym.components.observations import BaseObservations, OBSERVATION_MODULES
from gl_gym.components.parameters import PARAMETER_PROVIDERS, BaseParameterProvider
from gl_gym.configs.default_params import init_default_params
from gl_gym.configs.greenlight_parameters import TOMATO_PARAMETER_REGISTRY
from gl_gym.components.rewards import BaseReward, REWARDS_MODULES
from gl_gym.components.weather import BaseWeatherSampler, WeatherRepository, WEATHER_SAMPLERS
from gl_gym.core.types import RewardContext, StepContext, WeatherScenario
from gl_gym.models.GreenLight.utils import define_model
from gl_gym.environments.utils import init_state, load_weather_data

_DEFAULT_WEATHER_DIR = Path(__file__).resolve().parent.parent / "data" / "weather"

ObservationSpec = Union[
    str,                              # registry key
    type[BaseObservations],            # class
    BaseObservations,                  # pre-built instance
    Callable[["GreenLightEnv"], BaseObservations],  # factory
]

class GreenLightEnv(gym.Env):
    def __init__(
        self,
        num_params: int,                # number of model parameters
        nx: int,                        # number of states
        nu: int,                        # number of control inputs
        nd: int,                        # number of disturbances
        dt: float,                      # [s] time step for the underlying GreenLight solver
        u_min: List[float],             # min bound for the control inputs
        u_max: List[float],             # max for the control inputs
        delta_u_max: float,             # max change for the control inputs
        season_length: int,             # length of the growing season [days]
        pred_horizon: int,              # lookahead horizon for weather predictions/realizations [days]
        observation_modules: Iterable[ObservationSpec] | None,
        constraints: Dict[str, Any],            # constraints for the environment
        reward_fn: Union[str, type[BaseReward], BaseReward, Callable[[], BaseReward]],
        weather_scenario_sampler: Union[str, BaseWeatherSampler, type[BaseWeatherSampler], Callable[[], BaseWeatherSampler]],
        weather_scenario_sampler_kwargs: Dict[str, Any],
        weather_repository: WeatherRepository | None = None,
        weather_data_dir: str | Path | None = None,
        reward_kwargs: Dict[str, Any] = {},     # reward function arguments
        controlled_inputs: List[str] = ["uBoil", "uCO2", "uThScr", "uVent", "uLamp", "uBlScr"],
        normalize_actions: bool = True,
        parameter_provider: Union[str, type[BaseParameterProvider], BaseParameterProvider, Callable[[], BaseParameterProvider]] = "fixed",
        parameter_provider_kwargs: Dict[str, Any] = {},
        ) -> None:
        super(GreenLightEnv, self).__init__()

        if weather_repository is None:
            base_dir = Path(weather_data_dir) if weather_data_dir is not None else _DEFAULT_WEATHER_DIR
            weather_repository = WeatherRepository(
                weather_data_dir=base_dir,
                load_weather_data_fn=load_weather_data,
            )

        # arguments that are kept the same over various simulations
        self.c = 86400  # Class constant (seconds in a day)
        self.num_params = num_params
        self.nx = nx
        self.nu = nu
        self.nd = nd
        self.u_min = np.array(u_min, dtype=np.float32)
        self.u_max = np.array(u_max, dtype=np.float32)
        self.delta_u_max = self.u_max * delta_u_max
        self.dt = dt
        self.season_length = season_length
        self.N = int(self.season_length * self.c/self.dt)
        self.pred_horizon = pred_horizon
        self.Np = int(self.pred_horizon * self.c/self.dt)


        # initialise the observation and action spaces
        self.observation_modules = self._init_observations_modules(observation_modules)

        # Preferred: structured observation space
        self.observation_space = self._build_observation_space()
        self.action_scheme = NamedControlActionScheme(
            nu=self.nu,
            controlled_inputs=controlled_inputs,
            low=self.u_min,
            high=self.u_max,
            normalize_actions=normalize_actions,
        )
        self.weather_scenario_sampler = self._init_weather_sampler(weather_scenario_sampler, weather_scenario_sampler_kwargs)
        self.weather_repository = weather_repository
        self.action_space = self.action_scheme.action_space

        # initialise the model
        self.F = define_model(
            nx=self.nx,
            nu=self.nu,
            nd=self.nd,
            n_params=self.num_params,
            dt=self.dt,
        )

        # The lowerbound constraints
        self.constraints_low = np.array([
            constraints["co2_min"],
            constraints["temp_min"],
            constraints["rh_min"],
        ])

        # The upperbound constraints
        self.constraints_high = np.array([
            constraints["co2_max"],
            constraints["temp_max"],
            constraints["rh_max"],
        ])

        # Parameter initialization
        self.base_p = np.asarray(init_default_params(self.num_params), dtype=np.float64)
        self.parameter_registry = TOMATO_PARAMETER_REGISTRY

        self.parameter_provider = self._init_parameter_provider(
            parameter_provider=parameter_provider,
            parameter_provider_kwargs=parameter_provider_kwargs,
        )

        # keep a valid initial vector before first reset
        self.p = self.base_p.copy()

        # initialise the reward function
        self.reward_fn = self._init_reward(reward_fn, reward_kwargs)

    def _terminalState(self) -> bool:
        """
        Function that checks whether the simulation has reached a terminal state.
        Terminal states are reached when the simulation has reached the end of the growing season.
        """
        if self.timestep >= self.N:
            return True
        return False

    def _build_observation_space(self) -> Space:
        return spaces.Dict(OrderedDict(
            (module.key, module.space)
            for module in self.observation_modules
        ))

    def _init_observations_modules(
        self,
        observation_modules: Iterable[ObservationSpec],
    ) -> list[BaseObservations]:
        modules: list[BaseObservations] = []

        for spec in observation_modules:
            if isinstance(spec, str):
                modules.append(OBSERVATION_MODULES[spec](self))
            elif isinstance(spec, type) and issubclass(spec, BaseObservations):
                modules.append(spec(self))
            elif isinstance(spec, BaseObservations):
                modules.append(spec)
            elif callable(spec):
                modules.append(spec(self))
            else:
                raise TypeError(f"Unsupported observation spec: {spec!r}")

        return modules

    def _init_reward(self, reward, reward_kwargs):
        reward_kwargs = reward_kwargs or {}

        if isinstance(reward, str):
            reward_cls = REWARDS_MODULES[reward]
            return reward_cls(p=self.p, dt=self.dt, **reward_kwargs)
        elif isinstance(reward, type) and issubclass(reward, BaseReward):
            return reward(p=self.p, dt=self.dt, **reward_kwargs)
        elif isinstance(reward, BaseReward):
            return reward
        elif callable(reward):
            return reward(self, **reward_kwargs)
        else:
            raise TypeError(f"Unsupported reward spec: {reward!r}")

    def _init_weather_sampler(self, weather_sampler, weather_sampler_kwargs):
        kwargs = weather_sampler_kwargs or {}

        if isinstance(weather_sampler, str):
            cls = WEATHER_SAMPLERS[weather_sampler]
            return cls(**kwargs)
        elif isinstance(weather_sampler, BaseWeatherSampler):
            return weather_sampler
        elif isinstance(weather_sampler, type) and issubclass(weather_sampler, BaseWeatherSampler):
            return weather_sampler(**kwargs)
        elif callable(weather_sampler):
            return weather_sampler(**kwargs)
        else:
            raise TypeError(f"Unsupported weather_sampler spec: {weather_sampler!r}")

    def _init_parameter_provider(
        self,
        parameter_provider="fixed",
        parameter_provider_kwargs=None,
    ):
        kwargs = dict(parameter_provider_kwargs or {})

        if isinstance(parameter_provider, str):
            provider_cls = PARAMETER_PROVIDERS[parameter_provider]
            return provider_cls(
                env=self,
                base_p=self.base_p,
                registry=self.parameter_registry,
                **kwargs,
            )

        if isinstance(parameter_provider, type) and issubclass(parameter_provider, BaseParameterProvider):
            return parameter_provider(
                env=self,
                base_p=self.base_p,
                registry=self.parameter_registry,
                **kwargs,
            )

        if isinstance(parameter_provider, BaseParameterProvider):
            return parameter_provider

        if callable(parameter_provider):
            return parameter_provider(
                env=self,
                base_p=self.base_p,
                registry=self.parameter_registry,
                **kwargs,
            )

        raise TypeError(f"Unsupported parameter_provider spec: {parameter_provider!r}")

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, SupportsFloat, bool, bool, Dict[str, Any]]:
        self.x_prev = np.copy(self.x)

        # convert agent's action to full control input vector that model takes as input
        self.u = self.action_scheme.to_full_control_input(action)

        try:
            p_dyn = ca.vertcat(ca.DM(self.weather_data[self.timestep]), self.p)
            res = self.F(x0=ca.DM(self.x), u=ca.DM(self.u), p=p_dyn)
            self.x = res["xf"].full().flatten()

        except:
            print("Error in ODE approximation")
            self.truncated = True

        # update time
        self.day_of_year += (self.dt/self.c) % 365
        self.hour_of_day +=  (self.dt/3600)
        self.hour_of_day = self.hour_of_day % 24

        self.obs = self._get_obs()
        if self._terminalState():
            self.terminated = True

        # create reward context such that reward function does not rely on env
        ctx = RewardContext(
            t=self.timestep,
            dt=self.dt,
            Np=self.Np,
            x_prev=self.x_prev,
            x=self.x,
            u=self.u,
            p=self.p,
            d=self.weather_data,
            obs=self.obs,
            day_of_year=self.day_of_year,
            hour_of_day=self.hour_of_day,
            constraints_low=self.constraints_low,
            constraints_high=self.constraints_high,
        )

        reward, reward_info = self.reward_fn.compute_reward(ctx)
        info = self._get_info(reward_info)
        self.timestep += 1

        return (
                self.obs,
                reward, 
                self.terminated, 
                self.truncated,
                info
                )

    def _get_obs(self):
        ctx = StepContext(
            t=self.timestep,
            dt=self.dt,
            Np=self.Np,
            x_prev=self.x_prev,
            x=self.x,
            u=self.u,
            p=self.p,
            d=self.weather_data,
            hour_of_day=self.hour_of_day,
            day_of_year=self.day_of_year,
        )
        obs = {
            module.key: np.asarray(module.compute_obs(ctx), dtype=np.float32)
            for module in self.observation_modules
        }
        return obs

    def get_obs_names(self):
        """
        """
        obs_names = []
        for module in self.observation_modules:
            obs_names.extend(module.obs_names)
        return obs_names

    def _get_info(self, reward_info: Dict[str, Any]) -> Dict[str, Any]:
        return {
            **reward_info,
            "controls": self.u,
        }

    def _resolve_weather_scenario(self, options: Dict[str, Any] | None) -> WeatherScenario:
        if options is not None and "scenario" in options:
            return WeatherScenario(
                location=options["scenario"]["location"],
                growth_year=int(options["scenario"]["growth_year"]),
                start_day=int(options["scenario"]["start_day"]),
            )
        return self.weather_scenario_sampler.sample(self.np_random, options)

    def reset(
        self, seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed, options=options)

        scenario = self._resolve_weather_scenario(options)
        weather_data = self.weather_repository.load(
            location=scenario.location,
            growth_year=scenario.growth_year,
            start_day=scenario.start_day,
            season_length=self.season_length,
            pred_horizon=self.pred_horizon,
            dt=self.dt,
            nd=self.nd,
        )

        self.weather_data = weather_data
        self.location = scenario.location
        self.growth_year = scenario.growth_year
        self.start_day = scenario.start_day

        self.day_of_year = self.start_day
        self.hour_of_day = 0

        self.p = self.parameter_provider.sample(
            rng=self.np_random,
            scenario=scenario,
            options=options,
        )

        self.u  = self.action_scheme.reset_full_control_input()
        self.x = init_state(self.weather_data[0])
        self.x_prev = np.copy(self.x)
        self.timestep = 0
        self.obs = self._get_obs()

        self.terminated = False
        self.truncated = False
        return self.obs, {
            "scenario":
                {"location": self.location, "growth_year": self.growth_year, "start_day": self.start_day}
            }
