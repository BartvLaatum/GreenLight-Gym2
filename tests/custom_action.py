import unittest

import numpy as np

from gl_gym.environments.greenlight_env import GreenLightEnv
from RL.utils import load_env_params, build_env_kwargs

class TestCustomAction(unittest.TestCase):
    def setUp(self):
        env_kwargs = load_env_params("GreenLightEnv", "gl_gym/configs/envs/")
        env_kwargs, _ = build_env_kwargs(env_kwargs)
        self.env = GreenLightEnv(**env_kwargs)
        self.env.reset(seed=42)

    def test_action_space_shape(self):
        self.assertEqual(self.env.action_space.shape, (self.env.action_scheme.controlled_idx.shape[0],))

    def test_initial_action(self):
        action = self.env.action_space.sample()
        self.assertEqual(action.shape, (self.env.action_scheme.controlled_idx.shape[0],))
        self.assertTrue(np.all(action <= 1.0))
        self.assertTrue(np.all(action >= -1.0))

    def test_to_full_action(self):
        action = self.env.action_space.sample()
        full_action = self.env.action_scheme.to_full_control_input(action)

        self.assertEqual(full_action.shape, (self.env.nu,))
        # self.assertTrue(np.all(full_action == action))

    def test_normalize_actions(self):
        action = np.zeros(self.env.action_space.shape)
        denormalized_action = self.env.action_scheme._denormalize(action)
        self.assertEqual(denormalized_action.shape, (6,))
        self.assertTrue(np.all(denormalized_action == 0.5))


    def test_step(self):
        action = self.env.action_space.sample()
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.assertIsInstance(reward, (int, float))
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)

    def test_step_without_normalization(self):
        # Create an environment with normalize_actions=False
        env_kwargs = load_env_params(
            "GreenLightEnv", "gl_gym/configs/envs/"
        )
        env_kwargs, eval_scenarios = build_env_kwargs(env_kwargs)
        env_kwargs["normalize_actions"] = False
        env = GreenLightEnv(**env_kwargs)
        env.reset(seed=123)
        # Action space is (low, high), not normalized to [-1, 1] but to actual bounds
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        self.assertIsInstance(reward, (int, float))
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)

if __name__ == "__main__":
    unittest.main()