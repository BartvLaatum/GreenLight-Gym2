import unittest
import numpy as np
from gl_gym.environments.tomato_env import TomatoEnv
from gl_gym.RL.utils import load_env_params

class TestTomatoEnv(unittest.TestCase):
    def setUp(self):
        # Set up environment parameters
        self.env_id = "TomatoEnv"
        self.env_config_path = "gl_gym/configs/envs/"
        self.env_base_params, self.env_specific_params = load_env_params(self.env_id, self.env_config_path)

        # Initialize environment
        self.env = TomatoEnv(base_env_params=self.env_base_params, **self.env_specific_params)
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
        self.assertEqual(len(obs), self.env.observation_space.shape[0])

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
        self.assertEqual(len(obs), self.env.observation_space.shape[0])

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
    
