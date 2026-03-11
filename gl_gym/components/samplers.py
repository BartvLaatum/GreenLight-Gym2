from dataclasses import dataclass
import numpy as np

from gl_gym.core.types import Scenario

@dataclass(frozen=True)
class FixedScenarioSampler:
    scenario: Scenario

    def sample(self, rng: np.random.Generator, options=None) -> Scenario:
        return self.scenario


@dataclass
class FixedParameterProvider:
    p: np.ndarray

    def sample(self, rng: np.random.Generator, scenario: Scenario, options=None) -> np.ndarray:
        return self.p.copy()


@dataclass
class FixedInitialStateProvider:
    x0: np.ndarray

    def sample(self, rng: np.random.Generator, scenario: Scenario, p: np.ndarray, options=None) -> np.ndarray:
        return self.x0.copy()