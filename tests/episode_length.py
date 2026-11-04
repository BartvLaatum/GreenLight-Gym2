import unittest

import gymnasium as gym
import numpy as np

import gl_gym  # registers the envs

DEFAULT_MODULES = [
    "BasicCropObservations",
    "ControlObservations",
    "IndoorClimateObservations",
    "WeatherObservations",
    "TimeObservations",
]
TEMP_COL = 1  # outdoor temperature passes through WeatherObservations unconverted


class TestEpisodeLength(unittest.TestCase):
    def make_env(self, **kwargs):
        return gym.make("gl_gym/GreenLightTomato-v0", season_length=1, **kwargs).unwrapped

    def run_episode(self, **kwargs):
        env = self.make_env(**kwargs)
        env.reset(seed=0)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        steps, terminated = 0, False
        while not terminated and steps <= env.N:
            _, _, terminated, _, _ = env.step(action)
            steps += 1
        return env, steps, terminated

    def test_episode_covers_exactly_the_season(self):
        """season_length=1 at dt=900 is N=96 intervals, so the episode is 96 steps"""
        env, steps, terminated = self.run_episode()
        self.assertTrue(terminated)
        self.assertEqual(env.N, 96)
        self.assertEqual(steps, env.N)

    def test_weather_buffer_covers_the_forecast_horizon(self):
        """the last step observes d[N] and forecasts to d[N + Np], so N + Np + 1 rows"""
        env = self.make_env()
        env.reset(seed=0)
        self.assertEqual(env.weather_data.shape[0], env.N + env.Np + 1)

    def test_forecast_module_completes_an_episode(self):
        """WeatherForecastObservations reads d[t + Np] and must stay inside the buffer"""
        env, steps, terminated = self.run_episode(
            observation_modules=DEFAULT_MODULES + ["WeatherForecastObservations"],
        )
        self.assertTrue(terminated)
        self.assertEqual(steps, env.N)

    def test_observed_weather_drives_the_next_transition(self):
        """obs reports the row the upcoming step integrates, not the one already used"""
        env = self.make_env()
        obs, _ = env.reset(seed=0)
        d = env.weather_data
        action = np.zeros(env.action_space.shape, dtype=np.float32)

        # at reset the first step integrates d[0]
        self.assertEqual(env.timestep, 0)
        np.testing.assert_allclose(
            obs["WeatherObservations"][TEMP_COL], np.float32(d[0, TEMP_COL]), rtol=1e-6
        )

        for _ in range(40):
            obs, _, _, _, _ = env.step(action)
            # timestep is the row the next step() will integrate with
            np.testing.assert_allclose(
                obs["WeatherObservations"][TEMP_COL],
                np.float32(d[env.timestep, TEMP_COL]),
                rtol=1e-6,
            )

    def test_forecast_starts_strictly_in_the_future(self):
        """forecast[0] is the step after the current one, so it must not repeat the obs"""
        env = self.make_env(
            observation_modules=DEFAULT_MODULES + ["WeatherForecastObservations"],
        )
        env.reset(seed=0)
        d = env.weather_data
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        for _ in range(40):
            obs, _, _, _, _ = env.step(action)
        forecast = obs["WeatherForecastObservations"].reshape(env.Np, 5)
        t = env.timestep
        np.testing.assert_allclose(
            forecast[0][TEMP_COL], np.float32(d[t + 1, TEMP_COL]), rtol=1e-6
        )
        np.testing.assert_allclose(
            forecast[-1][TEMP_COL], np.float32(d[t + env.Np, TEMP_COL]), rtol=1e-6
        )


if __name__ == '__main__':
    unittest.main()
