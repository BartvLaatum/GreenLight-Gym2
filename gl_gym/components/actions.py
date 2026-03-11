from __future__ import annotations

import gymnasium as gym
import numpy as np

class NamedControlActionScheme:
    """
    Agent controls only a subset of the full greenhouse input vector.
    """

    def __init__(
        self,
        nu: int,                        # size of control input vector for the model (nu)
        controlled_inputs: list[str],   # list of controlled inputs
        low: np.ndarray,               # lower bound for the control inputs
        high: np.ndarray,              # upper bound for the control inputs
        normalize_actions: bool = True, # whether to normalize the actions
        delta_u_max: float = 0.1,      # maximum change for the control inputs
    ) -> None:

        control_defs = {
            "uBoil": 0,
            "uCO2": 1,
            "uThScr": 2,
            "uVent": 3,
            "uLamp": 4,
            "uBlScr": 5,
        }

        self.delta_u_max = delta_u_max
        # Check that all controlled_inputs are in control_defs
        assert all(c in control_defs for c in controlled_inputs), (
            f"All controlled_inputs must be among {list(control_defs.keys())}, got {controlled_inputs}"
        )

        self.nu = nu
        self.controlled_idx = np.asarray([control_defs[c] for c in controlled_inputs], dtype=int)
        self.low = np.asarray(low, dtype=np.float32)[self.controlled_idx]
        self.high = np.asarray(high, dtype=np.float32)[self.controlled_idx]

        self.normalize_actions = normalize_actions
        if self.normalize_actions:
            self._action_space = gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(len(self.controlled_idx),),
                dtype=np.float32,
            )
        else:
            self._action_space = gym.spaces.Box(
                low=self.low,
                high=self.high,
                shape=(len(self.controlled_idx),),
                dtype=np.float32,
            )

    def _initial_full_control_input(self) -> np.ndarray:
        u0 = np.zeros(self.nu, dtype=np.float32)
        u0[self.controlled_idx] = 0.5
        return u0

    @property
    def action_space(self) -> gym.Space:
        return self._action_space

    def _action_to_control(self, action: np.ndarray) -> np.ndarray:
        """
        Function that converts the action to control inputs.
        """
        return np.clip(self._prev_u_full[self.controlled_idx] + action*self.delta_u_max, self.low, self.high) 

    def _denormalize(self, action: np.ndarray) -> np.ndarray:
        # map [-1, 1] -> [low, high]
        return self.low + 0.5 * (action + 1.0) * (self.high - self.low)

    def to_full_control_input(
        self,
        action: np.ndarray,
    ) -> np.ndarray:
        prev_u_full = self._prev_u_full
        controlled_idx = self.controlled_idx

        if self.normalize_actions:
            controlled_action = self._action_to_control(action)
        else:
            controlled_action = action

        prev_u_full[controlled_idx] = controlled_action
        return prev_u_full.copy()

    def reset_full_control_input(self) -> np.ndarray:
        self._prev_u_full = self._initial_full_control_input()

        return self._prev_u_full.copy()