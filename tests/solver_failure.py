import unittest

import gymnasium as gym
import numpy as np

import gl_gym  # registers the envs


def failing_integrator(exc):
    def F(**kwargs):
        raise exc
    return F


class TestSolverFailure(unittest.TestCase):
    def setUp(self):
        self.env = gym.make("gl_gym/GreenLightTomato-v0", season_length=1).unwrapped
        self.env.reset(seed=0)
        self.action = np.zeros(self.env.action_space.shape, dtype=np.float32)

    def test_normal_step_reports_no_solver_failure(self):
        _, _, _, truncated, info = self.env.step(self.action)
        self.assertFalse(truncated)
        self.assertFalse(info["solver_failed"])

    def test_solver_error_truncates_and_is_reported(self):
        """The casadi error is passed on in a warning and flagged in info"""
        self.env.F = failing_integrator(RuntimeError("Error in Function::call for 'F'"))
        with self.assertWarnsRegex(RuntimeWarning, "Error in Function::call for 'F'"):
            _, _, _, truncated, info = self.env.step(self.action)
        self.assertTrue(truncated)
        self.assertTrue(info["solver_failed"])

    def test_keyboard_interrupt_is_not_swallowed(self):
        """Ctrl-C during integration should stop the run, not truncate the episode"""
        self.env.F = failing_integrator(KeyboardInterrupt())
        with self.assertRaises(KeyboardInterrupt):
            self.env.step(self.action)


if __name__ == '__main__':
    unittest.main()
