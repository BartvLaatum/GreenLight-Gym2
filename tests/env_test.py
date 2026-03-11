import unittest
import numpy as np
from gl_gym.environments.greenlight_env import GreenLightEnv
from RL.utils import load_env_params, build_env_kwargs

class TestGreenLightEnv(unittest.TestCase):
    def setUp(self):
        # Set up environment parameters
        self.env_id = "GreenLightEnv"
        self.env_config_path = "configs/envs/"
        self.env_kwargs = load_env_params(self.env_id, self.env_config_path)
        self.env_kwargs, _ = build_env_kwargs(self.env_kwargs)

        # Initialize environment
        self.env = GreenLightEnv(**self.env_kwargs)
        self.env.reset(seed=42)

    def test_reward_normalisation(self):
        """Test environment reset functionality"""
        obs, info = self.env.reset(seed=42)
        max_reward = 0.328 * 900 * 1e-6 / 0.065 * 1.6
        self.assertAlmostEqual(self.env.reward.max_profit, max_reward)
        self.assertEqual(self.env.reward.variable_costs, 0)
        # Check observation space
        action = np.ones(self.env.nu)*1
        self.env.u = np.ones(self.env.nu) * 1
        obs, reward, terminated, truncated, info = self.env.step(action)
        print(self.env.reward.scale_reward(self.env.reward.profit, self.env.reward.min_profit, self.env.reward.max_profit))

        violations = self.env.reward.output_violations()
        scaled_violations = self.env.reward.scale_reward(violations, self.env.reward.min_state_violations, self.env.reward.max_state_violations)

    def test_reset(self):
        """Test environment reset functionality"""
        obs, info = self.env.reset(seed=42)
        # Check observation space
        total_obs_size = sum(
            np.prod(space.shape) if hasattr(space, "shape") and space.shape is not None else 1
            for space in self.env.observation_space.spaces.values()
        )

        self.assertEqual(len(obs), total_obs_size)

        # Check initial state
        self.assertEqual(self.env.timestep, 0)
        self.assertFalse(self.env.terminated)

    def test_step(self):
        """Test environment step functionality"""
        self.env.reset()
        
        # Take a random action
        action = self.env.action_space.sample()
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Check observation
        total_obs_size = sum(
            np.prod(space.shape) if hasattr(space, "shape") and space.shape is not None else 1
            for space in self.env.observation_space.spaces.values()
        )
        self.assertEqual(len(obs), total_obs_size)

        # Check reward is float
        self.assertIsInstance(reward, (int, float))
        
        # Check timestep increment
        self.assertEqual(self.env.timestep, 1)

    def test_reward(self):
        """Test reward functionality"""
        self.env.reset()
        action = np.ones(self.env.nu)*-1
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.assertIsInstance(reward, (int, float))
        self.assertEqual(self.env.reward.variable_costs, 0)
        

    def test_action_scaling(self):
        """Test action scaling functionality"""
        action = self.env.action_space.sample()
        scaled_action = self.env.action_to_control(action)

        # Check scaled action bounds
        self.assertTrue(np.all(scaled_action >= self.env.u_min))
        self.assertTrue(np.all(scaled_action <= self.env.u_max))

    def test_episode_termination(self):
        """Test if episode terminates correctly"""
        self.env.reset()
        
        # Run until termination
        terminated = False
        steps = 0
        while not terminated and steps < self.env_base_params["season_length"] * 86400 // self.env_base_params["dt"] + 1:
            action = self.env.action_space.sample()
            _, _, terminated, _, _ = self.env.step(action)
            steps += 1

        # Check if terminated at correct step
        expected_steps = self.env_base_params["season_length"] * 86400 // self.env_base_params["dt"] + 1
        self.assertEqual(steps, expected_steps)
        self.assertTrue(terminated)

if __name__ == '__main__':
    unittest.main()
    
