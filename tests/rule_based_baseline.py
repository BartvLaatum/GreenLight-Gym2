import unittest

import numpy as np

from gl_gym.environments.greenlight_env import GreenLightEnv
from gl_gym.components.rule_based import RuleBasedController
from gl_gym.components.actions import NamedControlActionScheme
from gl_gym.core.types import StepContext
from RL.utils import load_env_params, build_env_kwargs

# Canonical ordering used by both the rule-based controller and the action scheme
ALL_CONTROL_NAMES = ["uBoil", "uCO2", "uThScr", "uVent", "uLamp", "uBlScr"]

# Default rule-based controller settings (from configs/agents/rule_based.yml)
RB_PARAMS = dict(
    lamps_on=0, lamps_off=18, lamps_day_start=-1, lamps_day_stop=366,
    lamps_off_sun=400, lamp_rad_sum_limit=10,
    temp_setpoint_day=19.5, temp_setpoint_night=16.5,
    heat_correction=0, heat_deadzone=5,
    co2_day=800, vent_heat_Pband=4,
    rh_max=85, mech_dehumid_Pband=2, vent_rh_Pband=5,
    t_vent_off=1, vent_cold_Pband=-1,
    thScrSpDay=5, thScrSpNight=10, thScrPband=-1, thScrDeadZone=4,
    thScrRh=-2, thScrRhPband=2,
    lampExtraHeat=2, blScrExtraRh=100, rhMax=85,
    tHeatBand=-1, co2Band=-100, useBlScr=1,
)


class TestRuleBasedBaseline(unittest.TestCase):
    """
    Verify that the RuleBasedController output is consistent with
    the NamedControlActionScheme mapping, and that subsets of
    controlled_inputs only touch the intended indices.
    """

    def setUp(self):
        self.env_kwargs = load_env_params("GreenLightEnv", "gl_gym/configs/envs/")
        self.env_kwargs, _ = build_env_kwargs(self.env_kwargs)
        self.controller = RuleBasedController(**RB_PARAMS)

    def _make_env(self, controlled_inputs=None, normalize_actions=False):
        kwargs = self.env_kwargs.copy()
        if controlled_inputs is not None:
            kwargs["controlled_inputs"] = controlled_inputs
        kwargs["normalize_actions"] = normalize_actions
        return GreenLightEnv(**kwargs)

    def _build_step_context(self, env):
        return StepContext(
            t=env.timestep,
            dt=env.dt,
            Np=env.Np,
            x_prev=env.x_prev,
            x=env.x,
            u=env.u,
            p=env.p,
            d=env.weather_data,
            hour_of_day=env.hour_of_day,
            day_of_year=env.day_of_year,
        )

    # ------------------------------------------------------------------
    # Index-mapping consistency
    # ------------------------------------------------------------------
    def test_action_scheme_maps_all_controls_to_sequential_indices(self):
        """With all 6 controls, controlled_idx should be [0, 1, 2, 3, 4, 5]."""
        env = self._make_env(controlled_inputs=ALL_CONTROL_NAMES)
        np.testing.assert_array_equal(
            env.action_scheme.controlled_idx,
            np.arange(len(ALL_CONTROL_NAMES)),
        )

    def test_action_scheme_maps_subset_to_expected_indices(self):
        """A subset should map to the correct sparse indices."""
        subset = ["uBoil", "uVent", "uLamp"]
        expected_idx = [0, 3, 4]
        env = self._make_env(controlled_inputs=subset)
        np.testing.assert_array_equal(
            env.action_scheme.controlled_idx,
            expected_idx,
        )

    # ------------------------------------------------------------------
    # Rule-based controller output properties
    # ------------------------------------------------------------------
    def test_controller_output_shape(self):
        """predict() must return a vector of length nu."""
        env = self._make_env()
        env.reset(seed=0)
        ctx = self._build_step_context(env)
        u = self.controller.predict(ctx)
        self.assertEqual(u.shape, (env.nu,))

    def test_controller_output_within_unit_interval(self):
        """All control values should be in [0, 1]."""
        env = self._make_env()
        env.reset(seed=0)
        ctx = self._build_step_context(env)
        u = self.controller.predict(ctx)
        self.assertTrue(np.all(u >= 0.0), f"Negative control value: {u}")
        self.assertTrue(np.all(u <= 1.0), f"Control value > 1: {u}")

    # ------------------------------------------------------------------
    # Feeding rule-based output through NamedControlActionScheme
    # ------------------------------------------------------------------
    def test_full_controls_placed_at_correct_indices(self):
        """
        With all controls active and normalize_actions=False, feeding
        the rule-based output through the action scheme should reproduce
        the same values at every index.
        """
        env = self._make_env(
            controlled_inputs=ALL_CONTROL_NAMES,
            normalize_actions=False,
        )
        env.reset(seed=0)
        ctx = self._build_step_context(env)

        u_rb = self.controller.predict(ctx)
        u_full = env.action_scheme.to_full_control_input(u_rb)

        np.testing.assert_allclose(u_full, u_rb, atol=1e-7)

    def test_subset_only_modifies_controlled_indices(self):
        """
        With a subset of controlled_inputs the action scheme should
        only write to those indices; the rest must stay at their
        initial value (0 for uncontrolled indices after reset).
        """
        subset = ["uBoil", "uVent"]
        expected_idx = np.array([0, 3])

        env = self._make_env(
            controlled_inputs=subset,
            normalize_actions=False,
        )
        env.reset(seed=0)

        # Capture the initial full control vector from reset
        u_initial = env.action_scheme.reset_full_control_input()

        ctx = self._build_step_context(env)
        u_rb = self.controller.predict(ctx)

        # Extract only the controlled values from the full controller output
        action = u_rb[expected_idx]
        u_full = env.action_scheme.to_full_control_input(action)

        # Controlled indices must carry the rule-based values
        np.testing.assert_allclose(
            u_full[expected_idx], u_rb[expected_idx], atol=1e-7,
            err_msg="Controlled indices do not match rule-based output",
        )

        # Uncontrolled indices must remain at their initial values
        all_idx = set(range(env.nu))
        uncontrolled_idx = sorted(all_idx - set(expected_idx.tolist()))

        np.testing.assert_allclose(
            u_full[uncontrolled_idx],
            u_initial[uncontrolled_idx],
            atol=1e-7,
            err_msg="Uncontrolled indices were modified by the action scheme",
        )

    def test_rule_based_values_arrive_at_named_controls(self):
        """
        For each control name, the value produced by the rule-based
        controller at its index must end up at the same index after
        going through the action scheme.
        """
        for name in ALL_CONTROL_NAMES:
            with self.subTest(control=name):
                env = self._make_env(
                    controlled_inputs=[name],
                    normalize_actions=False,
                )
                env.reset(seed=0)
                ctx = self._build_step_context(env)

                u_rb = self.controller.predict(ctx)
                idx = env.action_scheme.controlled_idx[0]

                action = np.array([u_rb[idx]], dtype=np.float32)
                u_full = env.action_scheme.to_full_control_input(action)

                self.assertAlmostEqual(
                    float(u_full[idx]),
                    float(u_rb[idx]),
                    places=5,
                    msg=f"{name}: action scheme placed value at wrong index",
                )


if __name__ == "__main__":
    unittest.main()
