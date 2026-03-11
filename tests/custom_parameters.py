import unittest

import numpy as np

from gl_gym.environments.greenlight_env import GreenLightEnv
from RL.utils import load_env_params, build_env_kwargs


class TestCustomParameters(unittest.TestCase):
    """
    Tests for the new parameter-provider structure.

    Assumptions:
    - GreenLightEnv exposes:
        - self.base_p
        - self.p
    - GreenLightEnv accepts:
        - parameter_provider
        - parameter_provider_kwargs
    - reset(options=...) supports:
        - parameter_overrides
    - the internal parameter registry contains the names used below
    """

    # Rename these if your registry uses different names
    SAMPLED_PARAMETER_NAMES = [
        "max_heating_power",
        "max_co2_dosing",
        "lamp_power",
    ]

    def setUp(self):
        self.env_kwargs = load_env_params("GreenLightEnv", "gl_gym/configs/envs/")
        self.env_kwargs, _ = build_env_kwargs(self.env_kwargs)

        # Build a nominal env first so we can inspect base_p and derive fixed test values
        self.nominal_env = GreenLightEnv(**self.env_kwargs)
        self.nominal_env.reset(seed=0)

        self.base_p = self.nominal_env.base_p.copy()

        # Assumed indices matching the registry used in the implementation example.
        # If your registry uses different indices/names, update these here.
        self.param_indices = {
            "max_heating_power": 108,
            "max_co2_dosing": 109,
            "lamp_power": 172,
        }

        # Deterministic values that are guaranteed to differ from the base vector
        self.fixed_sample_values = {
            "max_heating_power": float(self.base_p[self.param_indices["max_heating_power"]] * 1.10),
            "max_co2_dosing": float(self.base_p[self.param_indices["max_co2_dosing"]] * 0.90),
            "lamp_power": float(self.base_p[self.param_indices["lamp_power"]] * 1.05),
        }

    def _make_env(self, parameter_provider="fixed", parameter_provider_kwargs=None):
        env_kwargs = self.env_kwargs.copy()
        env_kwargs["parameter_provider"] = parameter_provider
        env_kwargs["parameter_provider_kwargs"] = parameter_provider_kwargs or {}
        return GreenLightEnv(**env_kwargs)

    def test_fixed_provider_returns_base_parameters(self):
        env = self._make_env(
            parameter_provider="fixed",
            parameter_provider_kwargs={},
        )

        env.reset(seed=123)

        self.assertEqual(env.p.shape, env.base_p.shape)
        np.testing.assert_allclose(env.p, env.base_p)

    def test_randomized_provider_changes_only_selected_parameters(self):
        env = self._make_env(
            parameter_provider="randomized",
            parameter_provider_kwargs={
                "sample_specs": {
                    name: {
                        "dist": "fixed",
                        "value": value,
                    }
                    for name, value in self.fixed_sample_values.items()
                }
            },
        )

        env.reset(seed=123)

        changed_indices = [self.param_indices[name] for name in self.SAMPLED_PARAMETER_NAMES]
        unchanged_indices = [
            i for i in range(env.base_p.shape[0]) if i not in changed_indices
        ]

        # Selected parameters should equal the requested fixed values
        for name in self.SAMPLED_PARAMETER_NAMES:
            idx = self.param_indices[name]
            self.assertAlmostEqual(float(env.p[idx]), float(self.fixed_sample_values[name]), places=10)

        # Everything else should remain equal to base_p
        np.testing.assert_allclose(
            env.p[unchanged_indices],
            env.base_p[unchanged_indices],
        )

    def test_randomized_provider_is_deterministic_for_same_seed(self):
        env = self._make_env(
            parameter_provider="randomized",
            parameter_provider_kwargs={
                "sample_specs": {
                    "max_heating_power": {
                        "dist": "relative_uniform",
                        "low_frac": 0.9,
                        "high_frac": 1.1,
                    },
                    "max_co2_dosing": {
                        "dist": "relative_uniform",
                        "low_frac": 0.85,
                        "high_frac": 1.15,
                    },
                    "lamp_power": {
                        "dist": "relative_uniform",
                        "low_frac": 0.95,
                        "high_frac": 1.05,
                    },
                }
            },
        )

        env.reset(seed=42)
        p1 = env.p.copy()

        env.reset(seed=42)
        p2 = env.p.copy()

        np.testing.assert_allclose(p1, p2)

    def test_randomized_provider_changes_at_least_one_selected_parameter(self):
        env = self._make_env(
            parameter_provider="randomized",
            parameter_provider_kwargs={
                "sample_specs": {
                    "max_heating_power": {
                        "dist": "relative_uniform",
                        "low_frac": 0.9,
                        "high_frac": 1.1,
                    },
                    "max_co2_dosing": {
                        "dist": "relative_uniform",
                        "low_frac": 0.85,
                        "high_frac": 1.15,
                    },
                    "lamp_power": {
                        "dist": "relative_uniform",
                        "low_frac": 0.95,
                        "high_frac": 1.05,
                    },
                }
            },
        )

        env.reset(seed=7)

        changed_indices = [self.param_indices[name] for name in self.SAMPLED_PARAMETER_NAMES]

        # At least one selected parameter should differ from the base vector
        any_changed = any(
            not np.isclose(env.p[idx], env.base_p[idx])
            for idx in changed_indices
        )
        self.assertTrue(any_changed)

    def test_parameter_overrides_work_with_fixed_provider(self):
        env = self._make_env(
            parameter_provider="fixed",
            parameter_provider_kwargs={},
        )

        override_value = float(self.base_p[self.param_indices["lamp_power"]] * 1.25)

        env.reset(
            seed=0,
            options={
                "parameter_overrides": {
                    "lamp_power": override_value,
                }
            },
        )

        self.assertAlmostEqual(
            float(env.p[self.param_indices["lamp_power"]]),
            override_value,
            places=10,
        )

        # Other parameters should remain identical to base_p
        unchanged_indices = [
            i for i in range(env.base_p.shape[0]) if i != self.param_indices["lamp_power"]
        ]
        np.testing.assert_allclose(
            env.p[unchanged_indices],
            env.base_p[unchanged_indices],
        )

    def test_parameter_overrides_work_with_randomized_provider(self):
        env = self._make_env(
            parameter_provider="randomized",
            parameter_provider_kwargs={
                "sample_specs": {
                    "max_heating_power": {
                        "dist": "fixed",
                        "value": self.fixed_sample_values["max_heating_power"],
                    },
                    "max_co2_dosing": {
                        "dist": "fixed",
                        "value": self.fixed_sample_values["max_co2_dosing"],
                    },
                    "lamp_power": {
                        "dist": "fixed",
                        "value": self.fixed_sample_values["lamp_power"],
                    },
                }
            },
        )

        override_value = float(self.base_p[self.param_indices["lamp_power"]] * 1.50)

        env.reset(
            seed=0,
            options={
                "parameter_overrides": {
                    "lamp_power": override_value,
                }
            },
        )

        # Override should win over the sampled value
        self.assertAlmostEqual(
            float(env.p[self.param_indices["lamp_power"]]),
            override_value,
            places=10,
        )

        # The other sampled parameters should still match the fixed sample values
        self.assertAlmostEqual(
            float(env.p[self.param_indices["max_heating_power"]]),
            float(self.fixed_sample_values["max_heating_power"]),
            places=10,
        )

        self.assertAlmostEqual(
            float(env.p[self.param_indices["max_co2_dosing"]]),
            float(self.fixed_sample_values["max_co2_dosing"]),
            places=10,
        )

    def test_parameter_vector_shape_is_preserved(self):
        env = self._make_env(
            parameter_provider="randomized",
            parameter_provider_kwargs={
                "sample_specs": {
                    "max_heating_power": {
                        "dist": "relative_uniform",
                        "low_frac": 0.9,
                        "high_frac": 1.1,
                    }
                }
            },
        )

        env.reset(seed=0)

        self.assertEqual(env.p.shape, env.base_p.shape)
        self.assertEqual(env.p.ndim, 1)


if __name__ == "__main__":
    unittest.main()