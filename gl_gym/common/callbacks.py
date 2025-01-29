from stable_baselines3.common.callbacks import BaseCallback
from typing import Optional
import os
import json

import os
from typing import Optional, Union, List, Any

import json
import wandb
import numpy as np
import pandas as pd
import gymnasium as gym

from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import VecEnv, sync_envs_normalization

from gl_gym.common.evaluation import evaluate_policy
from gl_gym.common.results import Results

class CustomWandbCallback(EvalCallback):
    """
    Callback that logs specified data from RL model and environment to Tensorboard.
    Is a daughter class of EvalCallback from Stable Baselines3.
    Saves the best model according to the mean reward on a evaluation environment.
    """
    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        path_vec_env: Optional[str] = None,         # from where to load in VecNormalize
        name_vec_env: Optional[str] = None,         # name of the VecNormalize file
        callback_on_new_best = None,                # callback to call when a new best model is found
        run: Optional[Any] = None,                  # wandb run
        results: Optional[Results] = None,          # results class where results are stored
        verbose: int = 1,
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            log_path=log_path,
            best_model_save_path=best_model_save_path,
            deterministic=deterministic,
            callback_on_new_best=callback_on_new_best,
            verbose=verbose,
        )
        self.path_vec_env = path_vec_env
        self.name_vec_env = name_vec_env
        self.run = run
        self.plot = True if run else False
        self.results = results
        self.save_results = True if results else False
        self.cum_metrics = {
            "EPI": 0,
            "revenue": 0.0,
            "variable_costs": 0.0,
            "fixed_costs": 0.0,
            "co2_cost": 0.0,
            "heat_cost": 0.0,
            "elec_cost": 0.0,
            "temp_violation": 0.0,
            "co2_violation": 0.0,
            "rh_violation": 0.0,
        }


        if self.save_results:
            self.results_path = f"data/{self.run.project}/{self.run.group}"
            # create save directory if not already present
            os.makedirs(self.results_path, exist_ok=True)

    def _cost_metrics_callback(self, local_vars, global_vars):
        # Extract the 'infos' variable from local_vars. 
        # 'infos' should be a list of info dicts returned by env.step().
        if "infos" not in local_vars:
            return

        infos = local_vars["infos"]

        # Accumulate metrics from infos
        # Assuming each info dictionary may contain keys like:
        # "revenue", "heating_cost", "co2_cost", "temperature_violation", 
        # "co2_violation", and "relative_humidity_violation".
        for info in infos:
            self.cum_metrics["EPI"] += info["EPI"]
            self.cum_metrics["revenue"] += info["revenue"]
            self.cum_metrics["lamp_violation"] += info["lamp_violation"]
            self.cum_metrics["temp_violation"] += info["temp_violation"]
            self.cum_metrics["co2_violation"] += info["co2_violation"]
            self.cum_metrics["rh_violation"] += info["rh_violation"]
            self.cum_metrics["variable_costs"] += info["variable_costs"]
            self.cum_metrics["fixed_costs"] += info["fixed_costs"]
            self.cum_metrics["co2_cost"] += info["co2_cost"]
            self.cum_metrics["heat_cost"] += info["heat_cost"]
            self.cum_metrics["elec_cost"] += info["elec_cost"]

    def _on_step(self) -> bool:

        continue_training = True

        if self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e


            # Reset cumulative metrics
            self.cum_metrics = {
                "EPI": 0.0,
                "revenue": 0.0,
                "lamp_violation": 0.0,
                "temp_violation": 0.0,
                "co2_violation": 0.0,
                "rh_violation": 0.0,
                "variable_costs": 0.0,
                "fixed_costs": 0.0,
                "co2_cost": 0.0,
                "heat_cost": 0.0,
                "elec_cost": 0.0,
            }

            episode_rewards, episode_lengths, add_info = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._cost_metrics_callback,
            )

            # we cutoff the last observations because that already belongs to the reset of the next episode
            # episode_obs = episode_obs[:, :-1, :]
            # time_vec = time_vec[:, :-1]

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )
            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)


                

            if self.verbose >= 1:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                # print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")

            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            # self.logger.record("eval/mean_episode_length", float(np.mean(episode_lengths)))
            # self.logger.record("eval/mea_profit", float(np.mean(sum_profits)))

            self.logger.record("eval/EPI", np.mean(self.cum_metrics["EPI"]))
            self.logger.record("eval/revenue", np.mean(self.cum_metrics["revenue"]))
            self.logger.record("eval/temp_violation", np.mean(self.cum_metrics["temp_violation"]))
            self.logger.record("eval/co2_violation", np.mean(self.cum_metrics["co2_violation"]))
            self.logger.record("eval/rh_violation", np.mean(self.cum_metrics["rh_violation"]))
            self.logger.record("eval/variable_costs", np.mean(self.cum_metrics["variable_costs"]))
            self.logger.record("eval/fixed_costs", np.mean(self.cum_metrics["fixed_costs"]))
            self.logger.record("eval/co2_cost", np.mean(self.cum_metrics["co2_cost"]))
            self.logger.record("eval/heat_cost", np.mean(self.cum_metrics["heat_cost"]))
            self.logger.record("eval/elec_cost", np.mean(self.cum_metrics["elec_cost"]))

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = mean_reward

                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

                    obs_names = self.eval_env.env_method("get_obs_names")[0][:23]
                    obs_df = pd.DataFrame(add_info["observations"][0][:int(5*86400/900)], columns=obs_names)
                    table = wandb.Table(dataframe=obs_df)
                    # Create a linspace vector for x-axis
                    cols2plot = ["co2_air", "temp_air","rh_air",  "pipe_temp", "cFruit","uBoil", "uCo2", "uThScr", "uVent", "uLamp", "uBlScr"]
                    for col in cols2plot:
                        wandb.log(
                            {
                                f"plot_{col}_id": wandb.plot.line(
                                    table, "timestep", col, title=f"Plot of {col} over steps"
                                )
                            }
                        )

                # # update the results class with the results of the current episode
                # if self.results is not None:
                #     # times = np.array([time_vec[i, :] for i in range(time_vec.shape[0])])

                #     # add dimension to time_vec and episode_profits so that they can be concatenated to other arrays
                #     times = np.expand_dims(times, axis=-1)
                #     episode_profits = np.expand_dims(episode_profits, axis=-1)
                #     episode_cum_profits = np.cumsum(episode_profits, axis=1)

                #     episode_gains = np.expand_dims(episode_gains, axis=-1)
                #     episode_var_costs = np.expand_dims(episode_var_costs, axis=-1)
                #     episode_fixed_costs = np.expand_dims(episode_fixed_costs, axis=-1)

                #     episode_cum_gains = np.cumsum(episode_gains, axis=1)
                #     episode_cum_var_costs = np.cumsum(episode_var_costs, axis=1)
                #     episode_cum_fixed_costs = np.cumsum(episode_fixed_costs, axis=1)

                    # # concatenate the results of the current episode to the results of the previous episodes
                    # data = np.concatenate((times, episode_obs[:,:,:26], episode_actions, episode_profits, episode_cum_profits, episode_cum_fixed_costs, episode_cum_var_costs, episode_cum_gains), axis=2)
                    # self.results.update_result(data)

                    # # save results
                    # if self.save_results:
                    #     # save results to csv given the run name
                    #     self.results.save(os.path.join(self.results_path, f"{self.run.name}.csv"))

                # plot results of a single episode (usually the first one)

                # if self.plot:
                #     plot_episode = 0
                #     table = wandb.Table(dataframe=self.results.df[(self.results.df['episode'] == plot_episode) &  (self.results.df['Time'] != 0)])

                #     for col in self.results.col_names[1:-1]:
                #         wandb.log(
                #             {
                #                 f"plot_{col}_id": wandb.plot.line(
                #                     table, "Time", col, title=f"Plot of {col} over Time"
                #                 )
                #             }
                #         )

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training

class SaveVecNormalizeCallback(BaseCallback):
    """
    Callback for saving a VecNormalize wrapper every ``save_freq`` steps
    FROM STABLE BASELINES3 RLZOO LIBRARY: 
    https://github.com/DLR-RM/rl-baselines3-zoo/blob/f6b3ff70b13d2c2156b3e0faf9994c107c649c82/utils/callbacks.py#L55
    
    :param save_freq: (int)
    :param save_path: (str) Path to the folder where ``VecNormalize`` will be saved, as ``vecnormalize.pkl``
    :param name_prefix: (str) Common prefix to the saved ``VecNormalize``, if None (default)
        only one file will be kept.
    """
    def __init__(self, save_freq: int, save_path: str, name_prefix: Optional[str] = None, verbose: int = 0):
        super(SaveVecNormalizeCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        # The normalization statistics to save
        self.normalization_stats = {
            "mean": None,
            "var": None,}

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            if self.name_prefix is not None:
                path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps.pkl")
            else:
                path = os.path.join(self.save_path, "best_vecnormalize.pkl")
            if self.model.get_vec_normalize_env() is not None:
                # save the entire environment with the VecNormalize wrapper
                self.model.get_vec_normalize_env().save(path)

                self.normalization_stats.update({
                    "mean": self.model.get_vec_normalize_env().obs_rms.mean.tolist(),
                    "var": self.model.get_vec_normalize_env().obs_rms.var.tolist(),
                })
                # only save the normlization statistics
                with open(os.path.join(self.save_path, "norm_stats.json"), "w") as f:
                    json.dump(self.normalization_stats, f)
                if self.verbose > 1:
                    print("-----------------------")
                    print(f"Saving VecNormalize to {path}")
                    print("-----------------------")
        return True
