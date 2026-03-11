from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

@dataclass(frozen=True)
class ParameterDef:
    name: str
    index: int
    low: float | None = None
    high: float | None = None
    unit: str = ""

class ParameterRegistry:
    def __init__(self, params: list[ParameterDef]):
        self.params = list(params)
        self._by_name = {p.name: p for p in self.params}
        self._by_index = {p.index: p for p in self.params}

        if len(self._by_name) != len(self.params):
            raise ValueError("Duplicate parameter names in parameter registry.")
        if len(self._by_index) != len(self.params):
            raise ValueError("Duplicate parameter indices in parameter registry.")

    def __len__(self) -> int:
        return len(self.params)

    def get(self, name: str) -> ParameterDef:
        try:
            return self._by_name[name]
        except KeyError as e:
            valid = ", ".join(self._by_name.keys())
            raise KeyError(f"Unknown parameter '{name}'. Valid names: {valid}") from e

    def names(self) -> list[str]:
        return [p.name for p in self.params]

class BaseParameterProvider(ABC):
    def __init__(self, env, base_p: np.ndarray, registry: ParameterRegistry | None = None):
        self.env = env
        self.base_p = np.asarray(base_p, dtype=np.float64)
        self.registry = registry

    @abstractmethod
    def sample(
        self,
        rng: np.random.Generator,
        scenario: Any | None = None,
        options: dict[str, Any] | None = None,
    ) -> np.ndarray:
        ...

class FixedParameterProvider(BaseParameterProvider):
    def sample(self, rng, scenario=None, options=None) -> np.ndarray:
        p = self.base_p.copy()

        if options is not None and "parameter_overrides" in options:
            p = self._apply_overrides(p, options["parameter_overrides"])

        return p

    def _apply_overrides(self, p: np.ndarray, overrides: dict[str, float]) -> np.ndarray:
        if self.registry is None:
            raise ValueError("parameter_overrides require a ParameterRegistry.")
        p = p.copy()
        for name, value in overrides.items():
            param_def = self.registry.get(name)
            p[param_def.index] = float(value)
        return p

class RandomizedParameterProvider(BaseParameterProvider):
    """
    Randomize only the named subset of parameters.
    Unspecified parameters stay equal to base_p.
    """

    def __init__(
        self,
        env,
        base_p: np.ndarray,
        registry: ParameterRegistry,
        sample_specs: dict[str, dict[str, Any]],
    ):
        super().__init__(env=env, base_p=base_p, registry=registry)
        self.sample_specs = dict(sample_specs)

    def sample(self, rng, scenario=None, options=None) -> np.ndarray:
        p = self.base_p.copy()

        sample_specs = self.sample_specs
        if options is not None and "parameter_sampling" in options:
            sample_specs = options["parameter_sampling"]

        for name, spec in sample_specs.items():
            param_def = self.registry.get(name)
            base_value = p[param_def.index]
            p[param_def.index] = self._draw_value(
                rng=rng,
                base_value=base_value,
                spec=spec,
                param_def=param_def,
            )

        if options is not None and "parameter_overrides" in options:
            p = self._apply_overrides(p, options["parameter_overrides"])

        return p

    def _apply_overrides(self, p: np.ndarray, overrides: dict[str, float]) -> np.ndarray:
        p = p.copy()
        for name, value in overrides.items():
            param_def = self.registry.get(name)
            p[param_def.index] = float(value)
        return p

    def _draw_value(
        self,
        rng: np.random.Generator,
        base_value: float,
        spec: dict[str, Any],
        param_def: ParameterDef,
    ) -> float:
        dist = spec["dist"]

        if dist == "fixed":
            value = float(spec["value"])

        elif dist == "uniform":
            value = float(rng.uniform(spec["low"], spec["high"]))

        elif dist == "normal":
            value = float(rng.normal(spec["mean"], spec["std"]))


        elif dist == "relative_uniform":
            frac = float(rng.uniform(spec["low_frac"], spec["high_frac"]))
            value = float(base_value) * frac

        elif dist == "relative_normal":
            frac = float(rng.normal(spec["mean_frac"], spec["std_frac"]))
            value = float(base_value) * frac

        elif dist == "choice":
            value = float(rng.choice(spec["values"]))

        else:
            raise ValueError(
                f"Unknown distribution '{dist}' for parameter '{param_def.name}'."
            )

        if param_def.low is not None:
            value = max(value, param_def.low)
        if param_def.high is not None:
            value = min(value, param_def.high)

        return float(value)


class SetParameterProvider(BaseParameterProvider):
    """
    Deterministic provider for evaluation benchmarks.
    Can optionally choose which set through reset(options=...).
    """

    def __init__(
        self,
        env,
        base_p: np.ndarray,
        registry: ParameterRegistry | None,
        parameter_sets: list[np.ndarray],
        default_index: int = 0,
    ):
        super().__init__(env=env, base_p=base_p, registry=registry)
        self.parameter_sets = [np.asarray(p, dtype=np.float64) for p in parameter_sets]
        self.default_index = int(default_index)

        if not self.parameter_sets:
            raise ValueError("parameter_sets must not be empty.")

        for i, p in enumerate(self.parameter_sets):
            if p.shape != self.base_p.shape:
                raise ValueError(
                    f"parameter_sets[{i}] has shape {p.shape}, expected {self.base_p.shape}."
                )

    def sample(self, rng, scenario=None, options=None) -> np.ndarray:
        idx = self.default_index
        if options is not None and "parameter_set_index" in options:
            idx = int(options["parameter_set_index"])

        p = self.parameter_sets[idx].copy()

        if options is not None and "parameter_overrides" in options:
            if self.registry is None:
                raise ValueError("parameter_overrides require a ParameterRegistry.")
            for name, value in options["parameter_overrides"].items():
                param_def = self.registry.get(name)
                p[param_def.index] = float(value)
        return p

PARAMETER_PROVIDERS = {
    "fixed": FixedParameterProvider,
    "randomized": RandomizedParameterProvider,
    "set": SetParameterProvider,
}
