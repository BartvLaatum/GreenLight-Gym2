import unittest

import numpy as np
from gymnasium import spaces

from gl_gym.components.observations import BaseObservations
from gl_gym.environments.greenlight_env import GreenLightEnv
from RL.utils import load_env_params, make_training_vec_env, make_eval_vec_env
from gl_gym.environments.utils import load_weather_data
from gl_gym.components.weather import WeatherRepository

class TestWeatherSampling(unittest.TestCase):
    def setUp(self):
        env_kwargs = load_env_params(
            "GreenLightEnv", "gl_gym/configs/envs/"
        )
        env_kwargs["weather_scenario_sampler"] = "fixed"
        env_kwargs["weather_scenario_sampler_kwargs"] = {
            "location": "Amsterdam",
            "growth_year": 2020,
            "start_day": 1,
        }
        env_kwargs["weather_repository"] = WeatherRepository(
            weather_data_dir="gl_gym/data/weather/",
            load_weather_data_fn=load_weather_data,
        )
        
        self.env = GreenLightEnv(**env_kwargs)
        self.env.reset(seed=42)

    def test_weather_fixed_scenario_sampler(self):
        scenario = self.env.weather_scenario_sampler.sample(np.random.default_rng(0))
        self.assertEqual(scenario.location, "Amsterdam")
        self.assertEqual(scenario.growth_year, 2020)
        self.assertEqual(scenario.start_day, 1)
        self.env.reset(seed=0)
        self.assertEqual(self.env.location, "Amsterdam")
        self.assertEqual(self.env.growth_year, 2020)
        self.assertEqual(self.env.start_day, 1)
        self.assertEqual(self.env.weather_data.shape[-1], 10)
        weather_data_1 = self.env.weather_data.copy()
        self.env.reset(seed=0)
        self.assertEqual(self.env.location, "Amsterdam")
        self.assertEqual(self.env.growth_year, 2020)
        self.assertEqual(self.env.start_day, 1)
        self.assertEqual(self.env.weather_data.shape[-1], 10)
        self.assertTrue(np.allclose(weather_data_1, self.env.weather_data))

    def test_weather_random_scenario_sampler(self):
        env_kwargs = load_env_params(
            "GreenLightEnv", "gl_gym/configs/envs/"
        )
        env_kwargs["weather_scenario_sampler"] = "random"
        env_kwargs["weather_scenario_sampler_kwargs"] = {
            "locations": ["Amsterdam", "NewYork", "Beijing"],
            "growth_years": [2018, 2019, 2020],
            "start_days": range(1, 2),
        }
        env_kwargs["weather_repository"] = WeatherRepository(
            weather_data_dir="gl_gym/data/weather/",
            load_weather_data_fn=load_weather_data,
        )
        env = GreenLightEnv(**env_kwargs)
        env.reset(seed=123)
        scenario = env.weather_scenario_sampler.sample(np.random.default_rng(0))
        self.assertIn(scenario.location, ["Amsterdam", "NewYork", "Beijing"])
        self.assertIn(scenario.growth_year, [2018, 2019, 2020])
        self.assertIn(scenario.start_day, range(1, 200))
        env.reset(seed=0)
        self.assertEqual(env.location, scenario.location)
        self.assertEqual(env.growth_year, scenario.growth_year)
        self.assertEqual(env.start_day, scenario.start_day)
        self.assertEqual(env.weather_data.shape[-1], 10)
        weather_data_1 = env.weather_data.copy()
        env.reset(seed=0)
        self.assertEqual(env.location, scenario.location)
        self.assertEqual(env.growth_year, scenario.growth_year)
        self.assertEqual(env.start_day, scenario.start_day)
        self.assertEqual(env.weather_data.shape[-1], 10)
        self.assertTrue(np.allclose(weather_data_1, env.weather_data))

    def test_reset_with_scenario(self):
        """Test that the environment resets with a specific scenario.
        """
        options = {
            "scenario": {
                "location": "NewYork",
                "growth_year": 2019,
                "start_day": 2,
            }
        }
        self.env.reset(seed=0, options=options)
        self.assertEqual(self.env.location, "NewYork")
        self.assertEqual(self.env.growth_year, 2019)
        self.assertEqual(self.env.start_day, 2)
        self.assertEqual(self.env.weather_data.shape[-1], 10)
        weather_data_1 = self.env.weather_data.copy()
        self.env.reset(seed=0, options=options)
        self.assertEqual(self.env.location, "NewYork")
        self.assertEqual(self.env.growth_year, 2019)
        self.assertEqual(self.env.start_day, 2)
        self.assertEqual(self.env.weather_data.shape[-1], 10)
        self.assertTrue(np.allclose(weather_data_1, self.env.weather_data))
        self.env.reset(seed=0)
        self.assertEqual(self.env.location, "Amsterdam")
        self.assertEqual(self.env.growth_year, 2020)
        self.assertEqual(self.env.start_day, 1)
        self.assertEqual(self.env.weather_data.shape[-1], 10)
        self.assertFalse(np.allclose(weather_data_1, self.env.weather_data))

    def test_cycling_scenario_sampler(self):
        env_kwargs = load_env_params(
            "GreenLightEnv", "gl_gym/configs/envs/"
        )
        env_kwargs["weather_scenario_sampler"] = "cycling"
        env_kwargs["weather_scenario_sampler_kwargs"] = {
            "scenarios": 
                [dict(location="Amsterdam", growth_year=2020, start_day=1), 
                 dict(location="NewYork", growth_year=2019, start_day=2), 
                 dict(location="Beijing", growth_year=2018, start_day=3)]
        }
        env_kwargs["weather_repository"] = WeatherRepository(
            weather_data_dir="gl_gym/data/weather/",
            load_weather_data_fn=load_weather_data,
        )
        env = GreenLightEnv(**env_kwargs)
        env.reset(seed=123)
        # scenario = env.weather_scenario_sampler.sample(np.random.default_rng(0))
        self.assertEqual(env.location, "Amsterdam")
        self.assertEqual(env.growth_year, 2020)
        self.assertEqual(env.start_day, 1)
        env.reset(seed=0)
        self.assertEqual(env.location, "NewYork")
        self.assertEqual(env.growth_year, 2019)
        self.assertEqual(env.start_day, 2)
        self.assertEqual(env.weather_data.shape[-1], 10)
        weather_data_1 = env.weather_data.copy()

        env.reset(seed=0, options={"scenario_index": 1})
        self.assertEqual(env.location, "NewYork")
        self.assertEqual(env.growth_year, 2019)
        self.assertEqual(env.start_day, 2)
        self.assertEqual(env.weather_data.shape[-1], 10)
        self.assertTrue(np.allclose(weather_data_1, env.weather_data))
        env.reset(seed=0, options={"scenario_index": 2})
        self.assertEqual(env.location, "Beijing")
        self.assertEqual(env.growth_year, 2018)
        self.assertEqual(env.start_day, 3)
        self.assertEqual(env.weather_data.shape[-1], 10)


    def test_vectorized_weather_sampling(self):
        env_base_params, env_specific_params = load_env_params(
            "GreenLightEnv", "gl_gym/configs/envs/"
        )
        env_specific_params["weather_scenario_sampler"] = "random"
        env_specific_params["weather_scenario_sampler_kwargs"] = {
            "locations": ["Amsterdam"],
            "growth_years": [2018, 2019, 2020],
            "start_days": range(0, 3),
        }
        env_specific_params["weather_repository"] = WeatherRepository(
            weather_data_dir="gl_gym/data/weather/",
            load_weather_data_fn=load_weather_data,
        )
        env = make_training_vec_env(
            env_id="GreenLightEnv",
            env_base_params=env_base_params,
            env_specific_params=env_specific_params,
            seed=123,
            n_envs=4,
        )

        obs = env.reset()
        self.assertEqual(
            env.env_method("get_wrapper_attr", "location"),
            ["Amsterdam"] * 4,
        )        
        for year in env.env_method("get_wrapper_attr", "growth_year"):
            self.assertIn(year, [2018, 2019, 2020])
        for start_day in env.env_method("get_wrapper_attr", "start_day"):
            self.assertIn(start_day, [0, 1, 2])
        env.reset()

    def test_eval_env_vectorized_weather_sampling(self):
        env_base_params, env_specific_params = load_env_params(
            "GreenLightEnv", "gl_gym/configs/envs/"
        )
        env_specific_params["weather_repository"] = WeatherRepository(
            weather_data_dir="gl_gym/data/weather/",
            load_weather_data_fn=load_weather_data,
        )
        eval_scenarios = [
            dict(location="Amsterdam", growth_year=2018, start_day=1),
            dict(location="NewYork", growth_year=2019, start_day=2),
            dict(location="Beijing", growth_year=2020, start_day=3),
            dict(location="Amsterdam", growth_year=2018, start_day=1),
            dict(location="Amsterdam", growth_year=2018, start_day=2),
            dict(location="Amsterdam", growth_year=2018, start_day=3),
        ]

        eval_env = make_eval_vec_env(
            env_id="GreenLightEnv",
            env_base_params=env_base_params,
            env_specific_params=env_specific_params,
            seed=123,
            n_envs=6,
            eval_weather_scenarios=eval_scenarios,
        )
        obs = eval_env.reset()

        for i, env_sampler in enumerate(eval_env.env_method("get_wrapper_attr", "weather_scenario_sampler")):
            scenario = env_sampler.sample(np.random.default_rng(0))
            self.assertEqual(scenario.location, eval_scenarios[i]["location"])
            self.assertEqual(scenario.growth_year, eval_scenarios[i]["growth_year"])
            self.assertEqual(scenario.start_day, eval_scenarios[i]["start_day"])

if __name__ == "__main__":
    unittest.main()
