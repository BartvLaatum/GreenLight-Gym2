import unittest

import numpy as np
from gymnasium import spaces

from gl_gym.components.observations import BaseObservations
from gl_gym.core.types import StepContext
from gl_gym.environments.greenlight_env import GreenLightEnv
from RL.utils import load_env_params, build_env_kwargs

class FruitGrowthObservations(BaseObservations):
    @property
    def key(self) -> str:
        return "fruit_growth"

    @property
    def space(self) -> spaces.Box:
        return spaces.Box(
            low=np.array([-1e-4, -1e6], dtype=np.float32),
            high=np.array([1e6, 1e6], dtype=np.float32),
            dtype=np.float32,
        )

    def compute_obs(self, ctx: StepContext) -> np.ndarray:
        fruit_weight = float(ctx.x[25])
        growth_rate = float(ctx.x[25] - ctx.x_prev[25])
        return np.array([fruit_weight, growth_rate], dtype=np.float32)

class TestCustomObservations(unittest.TestCase):
    def setUp(self):
        env_kwargs = load_env_params(
            "GreenLightEnv", "gl_gym/configs/envs/"
        )
        env_kwargs["observation_modules"] = [
            "IndoorClimateObservations",
            FruitGrowthObservations,
        ]
        env_kwargs, _ = build_env_kwargs(env_kwargs)
        self.env = GreenLightEnv(**env_kwargs)
        self.env.reset(seed=42)

    def _expected_total_obs_size(self, obs_space) -> int:
        return sum(
            int(np.prod(space.shape)) if getattr(space, "shape", None) is not None else 1
            for space in obs_space.spaces.values()
        )

    def test_observation_space_shape(self):
        self.assertIsInstance(self.env.observation_space, spaces.Dict)

        expected_size = 4 + 2  # IndoorClimate(4) + FruitGrowth(2)
        total_obs_size = self._expected_total_obs_size(self.env.observation_space)

        self.assertEqual(total_obs_size, expected_size)
        self.assertIn("fruit_growth", self.env.observation_space.spaces)
        self.assertEqual(self.env.observation_space.spaces["fruit_growth"].shape, (2,))

    def test_reset_obs_matches_space(self):
        obs, _ = self.env.reset(seed=0)

        self.assertIsInstance(obs, dict)
        self.assertTrue(self.env.observation_space.contains(obs))

        total_obs_size = sum(
            int(np.asarray(v).size) for v in obs.values()
        )
        expected_total_obs_size = self._expected_total_obs_size(self.env.observation_space)

        self.assertEqual(total_obs_size, expected_total_obs_size)

    def test_step_obs_matches_space(self):
        self.env.reset(seed=0)
        action = self.env.action_space.sample()
        obs, _, _, _, _ = self.env.step(action)

        self.assertIsInstance(obs, dict)
        self.assertTrue(self.env.observation_space.contains(obs))

        total_obs_size = sum(
            int(np.asarray(v).size) for v in obs.values()
        )
        expected_total_obs_size = self._expected_total_obs_size(self.env.observation_space)

        self.assertEqual(total_obs_size, expected_total_obs_size)

    def test_custom_module_values(self):
        self.env.reset(seed=0)
        action = self.env.action_space.sample()
        obs, _, _, _, _ = self.env.step(action)

        self.assertIn("fruit_growth", obs)

        fruit_growth = obs["fruit_growth"]
        self.assertEqual(fruit_growth.shape, (2,))

        fruit_weight = fruit_growth[0]
        growth_rate = fruit_growth[1]

        self.assertAlmostEqual(
            float(fruit_weight),
            float(np.float32(self.env.x[25])),
            places=5,
        )
        self.assertAlmostEqual(
            float(growth_rate),
            float(np.float32(self.env.x[25] - self.env.x_prev[25])),
            places=5,
        )


if __name__ == "__main__":
    unittest.main()