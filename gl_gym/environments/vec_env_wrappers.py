import numpy as np

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn, VecEnvWrapper

class VecNoisyObs(VecEnvWrapper):
    def __init__(self, venv: VecEnv):
        super().__init__(venv=venv, observation_space=venv.observation_space)
        
    def reset(self):
        obs = self.venv.reset()
        return obs

    def step_async(self, actions: np.ndarray) -> None:
        self.venv.step_async(actions)
        
    def step_wait(self) -> VecEnvStepReturn:
        obs, reward, done, info = self.venv.step_wait()
        return obs, reward, done, info