import unittest
from typing import SupportsFloat, Dict

import numpy as np

from gl_gym.components.price_model import FixedPrice, DailyFruitPriceTrajectory, HourlyPriceTrajectory
from gl_gym.components.rewards import BaseReward
from gl_gym.core.types import RewardContext
from gl_gym.environments.greenlight_env import GreenLightEnv
from RL.utils import load_env_params, build_env_kwargs


class ConstantReward(BaseReward):
    """Simple custom reward used to test reward injection."""

    def __init__(self, constant_value: float = 1.23, p: np.ndarray | None = None, dt: int | None = None) -> None:
        self.p = p
        self.dt = dt
        self.constant_value = float(constant_value)

    def compute_reward(self, ctx: RewardContext) -> tuple[SupportsFloat, Dict[str, float]]:
        return self.constant_value, {"constant_value": self.constant_value}


class StateSumReward(BaseReward):
    """Custom reward based on the current state, to test env-dependent reward logic."""

    def __init__(self, state_indices: list[int], scale: float = 1.0, p: np.ndarray | None = None, dt: int | None = None) -> None:
        self.p = p
        self.dt = dt
        self.state_indices = state_indices
        self.scale = float(scale)

    def compute_reward(self, ctx: RewardContext) -> tuple[SupportsFloat, Dict[str, float]]:
        return float(np.sum(ctx.x[self.state_indices]) * self.scale), {"state_sum": float(np.sum(ctx.x[self.state_indices]) * self.scale)}

class TrajectoryReward(BaseReward):
    """Custom reward based on the trajectory of the state."""
    def __init__(self, daily_fruit_prices: dict[int, float], hourly_electricity_prices: dict[int, float], p: np.ndarray | None = None, dt: int | None = None) -> None:
        self.p = p
        self.dt = dt
        self.dmfm = 0.065
        self.fruit_price = DailyFruitPriceTrajectory(
            prices_by_day=daily_fruit_prices)
        self.electricity_price = HourlyPriceTrajectory(
            prices_by_hour=hourly_electricity_prices)
        
    def compute_reward(self, ctx: RewardContext) -> tuple[SupportsFloat, Dict[str, float]]:
        fruit_price = self.fruit_price.get_price(ctx)
        electricity_price = self.electricity_price.get_price(ctx)
        return float((ctx.x[0]-ctx.x_prev[0])/self.dmfm*1e-6 * fruit_price - ctx.u[0] * electricity_price), {"fruit_price": fruit_price, "electricity_price": electricity_price}

class TestCustomRewards(unittest.TestCase):
    def setUp(self):
        self.env_kwargs = load_env_params(
            "GreenLightEnv", "gl_gym/configs/envs/"
        )
        self.env_kwargs, _ = build_env_kwargs(self.env_kwargs)

    def _make_env(self, reward_spec, reward_kwargs):
        env_kwargs = self.env_kwargs.copy()
        env_kwargs["reward_fn"] = reward_spec
        env_kwargs["reward_kwargs"] = reward_kwargs
        return GreenLightEnv(
            **env_kwargs,
        )

    def test_builtin_reward_returns_scalar(self):
        """
        Check that the default built-in reward can be instantiated and that
        env.step returns a scalar reward.
        """
        env = self._make_env(
            reward_spec="GreenhouseReward",
            reward_kwargs={
                "elec_price": 0.25,
                "heating_price": 0.08,
                "co2_price": 0.12,
                "fruit_price": 2.5,
                "pen_lamp": 0.5,
                "dmfm": 0.0627,
            },
        )

        obs, info = env.reset(seed=0)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        self.assertIsInstance(reward, (float, np.floating))
        self.assertTrue(np.isfinite(reward))

    def test_builtin_reward_is_deterministic_after_reset(self):
        """
        With the same seed and same action, reward should be reproducible
        if the env itself is deterministic under fixed weather/initialization.
        """
        reward_kwargs = {
            "elec_price": 0.25,
            "heating_price": 0.08,
            "co2_price": 0.12,
            "fruit_price": 2.5,
            "pen_lamp": 0.5,
            "dmfm": 0.0627,
        }

        env = self._make_env("GreenhouseReward", reward_kwargs)

        obs, _ = env.reset(seed=123)
        action = np.asarray(env.action_space.sample())
        _, reward1, _, _, _ = env.step(action)

        obs, _ = env.reset(seed=123)
        _, reward2, _, _, _ = env.step(action)

        self.assertAlmostEqual(float(reward1), float(reward2), places=8)

    def test_custom_constant_reward_class_is_used(self):
        """
        Users should be able to pass a custom reward class.
        The returned step reward should match the custom class output exactly.
        """
        env = self._make_env(
            reward_spec=ConstantReward,
            reward_kwargs={"constant_value": 7.5},
        )

        env.reset(seed=0)
        action = env.action_space.sample()
        _, reward, _, _, _ = env.step(action)

        self.assertAlmostEqual(float(reward), 7.5, places=8)

    def test_custom_constant_reward_changes_with_parameter(self):
        """
        The custom reward kwargs should be forwarded correctly.
        """
        env_a = self._make_env(
            reward_spec=ConstantReward,
            reward_kwargs={"constant_value": 3.0},
        )
        env_b = self._make_env(
            reward_spec=ConstantReward,
            reward_kwargs={"constant_value": 9.0},
        )

        env_a.reset(seed=0)
        env_b.reset(seed=0)

        action_a = env_a.action_space.sample()
        action_b = np.array(action_a, copy=True)

        _, reward_a, _, _, _ = env_a.step(action_a)
        _, reward_b, _, _, _ = env_b.step(action_b)

        self.assertAlmostEqual(float(reward_a), 3.0, places=8)
        self.assertAlmostEqual(float(reward_b), 9.0, places=8)

    def test_custom_state_sum_reward_class_is_used(self):
        """
        Check that a custom reward can depend on env state.
        """
        env = self._make_env(
            reward_spec=StateSumReward,
            reward_kwargs={
                "state_indices": [0, 1, 2],
                "scale": 0.5,
            },
        )

        env.reset(seed=0)
        action = env.action_space.sample()
        _, reward, _, _, _ = env.step(action)

        expected = float(np.sum(env.x[[0, 1, 2]]) * 0.5)
        self.assertAlmostEqual(float(reward), expected, places=6)

    def test_custom_reward_instance_can_be_selected(self):
        """
        If your env supports passing an already constructed reward instance,
        this checks that path too.
        """
        reward_instance = ConstantReward(p=np.zeros(100), dt=1, constant_value=4.2)

        # env_kwargs, _ = build_env_kwargs(self.env_kwargs.copy())
        env_kwargs = self.env_kwargs.copy()
        env_kwargs["reward_fn"] = reward_instance

        env = GreenLightEnv(
            **env_kwargs,
        )

        # If your env rebinds env on the reward instance during init, this should work.
        env.reset(seed=0)
        action = env.action_space.sample()
        _, reward, _, _, _ = env.step(action)

        self.assertAlmostEqual(float(reward), 4.2, places=8)


    def test_custom_trajectory_reward_class_is_used(self):
        """
        Check that a custom reward can depend on the trajectory of the state.
        """
        env = self._make_env(
            reward_spec=TrajectoryReward,
            reward_kwargs={
                "daily_fruit_prices": {day: np.random.uniform(0.5, 2.5) for day in range(365)},
                "hourly_electricity_prices": {hour: np.random.uniform(0.05, 0.25) for hour in range(24)},
            },
        )

        env.reset(seed=0)
        action = env.action_space.sample()
        _, reward, _, _, info = env.step(action)
        # self.assertTrue(float(reward) > 0.0)
        self.assertIn("fruit_price", info)
        self.assertIn("electricity_price", info)
        fruit_price = info["fruit_price"]
        electricity_price = info["electricity_price"]
        self.assertTrue(fruit_price > 0.0 and electricity_price > 0.0)
        self.assertTrue(fruit_price < 2.5 and electricity_price < 0.25)
        env.reset(seed=0)
        action = env.action_space.sample()
        _, reward, _, _, info = env.step(action)
        # self.assertTrue(float(reward) > 0.0)
        self.assertAlmostEqual(info["fruit_price"], fruit_price, places=8)
        self.assertAlmostEqual(info["electricity_price"], electricity_price, places=8)


if __name__ == "__main__":
    unittest.main()