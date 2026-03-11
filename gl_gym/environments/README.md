## Environments

This folder defines the greenhouse simulation environments, their observation modules, reward functions, default parameters, the underlying CasADi model, and weather data.

- `base_env.py`: Defines the abstract base class `GreenLightEnv`.
- `baseline.py`: Defines the rule-based baseline controller.
- `noise.py`: Contains a function for randomly sampling tomato crop parameters.
- `observations.py`: Individual observation modules you can mix, configure them in the configuration file.
- `parameters.py`: Default greenhouse parameters (tuned for a Dutch Venlo greenhouse).
- `rewards.py`: Reward functions, possibility to create your own. Default is `GreenhouseReward`.
- `tomato_env.py`: A concrete example environment (`TomatoEnv`) built on `GreenLightEnv`.
- `utils.py`: Contains utility functions required by greenhouse systems.
- `models/`: Underlying CasADi model setup.
- `weather/`: Weather CSV files organized by location and year.

### Create your own environment

You can design a custom environment by subclassing `GreenLightEnv`. Use `TomatoEnv` as a reference for wiring the model, observations, rewards, action/observation spaces, and weather loading.

Minimal skeleton:

```python
from typing import Dict, Any, Optional, Tuple
import numpy as np
from gymnasium import spaces

from gl_gym.environments.base_env import GreenLightEnv
from gl_gym.environments.observations import BaseObservations, MyObservations
from gl_gym.environments.rewards import BaseReward, MyReward


class MyEnv(GreenLightEnv):
    def __init__(self, base_env_params: Dict[str, Any], reward_params: Dict[str, Any]):
        super().__init__(**base_env_params)
        # Define modules and reward
        self.observation_modules = [MyObservations(self)]
        self.observation_space = self._generate_observation_space()
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.nu,), dtype=np.float32)
        self.reward = MyReward(self, **reward_params)
        # TODO: setup your model (see TomatoEnv.define_model usage)

    def _init_observations(self, *_):
        return self.observation_modules

    def _generate_observation_space(self) -> spaces.Box:
        lows = [m.observation_space().low for m in self.observation_modules]
        highs = [m.observation_space().high for m in self.observation_modules]
        return spaces.Box(low=np.concatenate(lows), high=np.concatenate(highs), dtype=np.float32)

    def _init_rewards(self) -> BaseReward:
        return self.reward

    def _get_obs(self):
        return np.concatenate([m.compute_obs() for m in self.observation_modules])

    def _get_info(self) -> Dict[str, Any]:
        return {}

    def _terminalState(self) -> bool:
        return self.timestep >= self.N

    def step(self, action):
        # TODO: map action -> controls, advance dynamics, update obs/reward
        raise NotImplementedError

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        # TODO: set season/year/day, load weather, initialise states/controls
        raise NotImplementedError
```

### Create a custom observation module

Observation modules extend `BaseObservations`. They define their own bounds and how to compute a vector of features.

```python
import numpy as np
from gymnasium import spaces
from gl_gym.environments.observations import BaseObservations


class MyObservations(BaseObservations):
    def __init__(self, env):
        self.env = env
        self.obs_names = ["my_measure_1", "my_measure_2"]
        self.n_obs = len(self.obs_names)

    def observation_space(self):
        return spaces.Box(low=-1e-4, high=1e4, shape=(self.n_obs,), dtype=np.float32)

    def compute_obs(self) -> np.ndarray:
        # Example: pull any combination of model state, control, or weather
        return np.array([
            float(self.env.x[2]),          # temp_air
            float(self.env.weather_data[self.env.timestep, 0]),  # glob_rad
        ], dtype=np.float32)
```

To use a new module with `TomatoEnv`, add its class to the `OBSERVATION_MODULES` dict in `tomato_env.py` or your own `<custom_env.py>`, then list it by name in the env config `observation_modules` array.

### Create a custom reward

Rewards extend `BaseReward` and implement `compute_reward()`.

```python
from typing import SupportsFloat
from gl_gym.environments.rewards import BaseReward


class MyReward(BaseReward):
    def __init__(self, env, weight: float = 1.0):
        self.env = env
        self.weight = weight

    def compute_reward(self) -> SupportsFloat:
        # Example: encourage higher canopy temperature while penalizing lamp usage
        temp_term = float(self.env.x[4]) * self.weight
        lamp_pen = float(self.env.u[4]) * 0.1
        return temp_term - lamp_pen
```

To use a new reward with `TomatoEnv`, add it to the `REWARDS` dict in `tomato_env.py` or your own `<custom_env.py>`, then select it by name in the env config via `reward_function` and pass parameters under `reward_params`.

### Hooking everything up via configuration

Create or update a config file under `gl_gym/configs/envs/` and set your environment-specific fields.

Example (snippet):

```yaml
GreenLightEnv:
  weather_data_dir: gl_gym/environments/weather # path to weather data
  location: Amsterdam             # location of the recorded weather data
  num_params: 208                 # number of model parameters
  nx: 28                          # number of states
  nu: 6                           # number of control inputs
  nd: 10                          # number of weather disturbances
  dt: 900                         # [s] time step for the underlying GreenLight solver
  u_min: [0, 0, 0, 0, 0, 0]
  u_max: [1, 1, 1, 1, 1, 1]
  delta_u_max: 0.1                # max change rate in control inputs
  pred_horizon: 0.5               # [days] number of future weather predictions
  season_length: 60               # number of days to simulate
  start_train_year: 2010          # start year for training
  end_train_year: 2010            # end year for training
  start_train_day: 59             # start day of the year for training
  end_train_day: 59               # end day of the year for training  
  training: True                  # whether we are training or testing

MyEnv:
  reward_function: MyReward
  observation_modules: [IndoorClimateObservations, MyObservations]
  reward_params:
    weight: 0.5
```

Then run training with `--env_id MyEnv`. If you stay with `TomatoEnv`, keep `--env_id TomatoEnv` and just reference your custom modules/reward by name in that section of the YAML.

### Parameters, model, and weather

- `parameters.py` contains default parameters tuned for a Dutch Venlo greenhouse. Modify these to reflect your greenhouse and crop settings.
- `models/` defines the ODE model in CasADi to propagate dynamics.
- `weather/` contains weather CSV files organized as `<location>/<year>.csv`. See the top-level [`README.md`](./README.md) for the expected headers and units.
