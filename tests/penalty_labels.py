import unittest

import gymnasium as gym
import numpy as np

import gl_gym  # registers the envs
from gl_gym.components.rewards import GreenhouseReward
from gl_gym.configs.default_params import init_default_params
from gl_gym.core.types import RewardContext


class TestPenaltyLabels(unittest.TestCase):
    def setUp(self):
        reward_kwargs = dict(gym.spec("gl_gym/GreenLightTomato-v0").kwargs["reward_kwargs"])
        self.p = np.asarray(init_default_params(208), dtype=np.float64)
        self.reward_fn = GreenhouseReward(**reward_kwargs, dt=900, p=self.p)

    def ctx(self, co2, temp, rh):
        # bounds in the env's constraint order: co2 [ppm], temp [C], rh [%]
        return RewardContext(
            t=0, dt=900, Np=0, x_prev=np.zeros(28), x=np.zeros(28),
            u=np.zeros(6), p=self.p, d=np.zeros((1, 10)),
            hour_of_day=12.0, day_of_year=59.0,
            obs={"IndoorClimateObservations": np.array([co2, temp, rh, 40.0], dtype=np.float32)},
            constraints_low=np.array([300.0, 15.0, 50.0]),
            constraints_high=np.array([1600.0, 34.0, 85.0]),
        )

    def test_temp_violation_goes_to_temp_penalty(self):
        """2 C over the temp bound should only show up in temp_penalty"""
        _, info = self.reward_fn.compute_reward(self.ctx(co2=800.0, temp=36.0, rh=70.0))
        self.assertAlmostEqual(info["temp_penalty"], 2 / 15)
        self.assertEqual(info["co2_penalty"], 0.0)

    def test_co2_violation_goes_to_co2_penalty(self):
        """250 ppm over the co2 bound should only show up in co2_penalty"""
        _, info = self.reward_fn.compute_reward(self.ctx(co2=1850.0, temp=22.0, rh=70.0))
        self.assertAlmostEqual(info["co2_penalty"], 250 / 2500)
        self.assertEqual(info["temp_penalty"], 0.0)


if __name__ == '__main__':
    unittest.main()
