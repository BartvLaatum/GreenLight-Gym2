import unittest

import gymnasium as gym
import numpy as np

import gl_gym  # registers the envs
from gl_gym.components.rewards import GreenhouseReward
from gl_gym.configs.default_params import init_default_params
from gl_gym.core.types import RewardContext

LAMPS_ON = (0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
LAMPS_OFF = (0.0,) * 6


class TestLampPenalty(unittest.TestCase):
    def setUp(self):
        reward_kwargs = dict(gym.spec("gl_gym/GreenLightTomato-v0").kwargs["reward_kwargs"])
        self.pen_lamp = reward_kwargs["pen_lamp"]
        self.p = np.asarray(init_default_params(208), dtype=np.float64)
        self.reward_fn = GreenhouseReward(**reward_kwargs, dt=900, p=self.p)

    def ctx(self, u, hour):
        # indoor climate inside the bounds, so the lamp is the only penalty
        return RewardContext(
            t=0, dt=900, Np=0, x_prev=np.zeros(28), x=np.zeros(28),
            u=np.asarray(u, dtype=np.float64), p=self.p, d=np.zeros((1, 10)),
            hour_of_day=hour, day_of_year=59.0,
            obs={"IndoorClimateObservations": np.array([800.0, 22.0, 70.0, 40.0], dtype=np.float32)},
            constraints_low=np.array([300.0, 15.0, 50.0]),
            constraints_high=np.array([1600.0, 34.0, 85.0]),
        )

    def test_lamps_on_after_20_are_penalized(self):
        """Lamps on at 21:00 should cost pen_lamp compared to the same step at noon"""
        reward_day, _ = self.reward_fn.compute_reward(self.ctx(LAMPS_ON, hour=12.0))
        reward_night, info = self.reward_fn.compute_reward(self.ctx(LAMPS_ON, hour=21.0))
        self.assertEqual(info["lamp_penalty"], self.pen_lamp)
        self.assertEqual(info["control_penalty"], self.pen_lamp)
        self.assertAlmostEqual(reward_day - reward_night, self.pen_lamp)

    def test_lamps_off_after_20_are_not_penalized(self):
        _, info = self.reward_fn.compute_reward(self.ctx(LAMPS_OFF, hour=21.0))
        self.assertEqual(info["lamp_penalty"], 0.0)

    def test_violation_resets_on_the_next_step(self):
        """One night step with the lamps on shouldn't stick around"""
        self.reward_fn.compute_reward(self.ctx(LAMPS_ON, hour=21.0))
        _, info = self.reward_fn.compute_reward(self.ctx(LAMPS_OFF, hour=12.0))
        self.assertEqual(self.reward_fn.lamp_violation, 0)
        self.assertEqual(info["lamp_penalty"], 0.0)


if __name__ == '__main__':
    unittest.main()
