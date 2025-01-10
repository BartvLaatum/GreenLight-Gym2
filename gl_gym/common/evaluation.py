import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import gymnasium as gym

from stable_baselines3.common import type_aliases
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.vec_env import VecMonitor, VecEnv, is_vecenv_wrapped

def evaluate_policy(
    model: BasePolicy,
    env: VecEnv,
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = True,
    warn: bool = True,
    save_info: bool = False,
) -> Union[Tuple[float, float], Tuple[List[float], List[int], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate. This can be any object
        that implements a `predict` method, such as an RL algorithm (``BaseAlgorithm``)
        or policy (``BasePolicy``).
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []
    episode_profits = []
    episode_gains = []
    episode_var_costs = []
    episode_fixed_costs = []
    episode_actions = []
    episode_obs = []
    episode_cbuffer = []
    episode_fruit_weight = []
    episode_dli = []
    episode_weather = []
    time_vec = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")

    # timestep counter for each seperate environment
    timestep = 0
    observations = env.reset()
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    N = env.get_attr("N", indices=0)[0]
    nu = env.get_attr("n_setpoints", indices=0)[0]
    current_episode_profits = np.zeros((n_envs, N))
    current_episode_gains = np.zeros((n_envs, N))
    current_episode_var_costs = np.zeros((n_envs, N))
    current_episode_fixed_costs = np.zeros((n_envs, N))
    current_episode_actions = np.zeros((n_envs, N, nu))
    current_episode_cbuffer = np.zeros((n_envs, N))
    current_episode_fruit_weight = np.zeros((n_envs, N))
    current_episode_dli = np.zeros((n_envs, N))
    current_episode_obs = np.zeros((n_envs, N, env.observation_space.shape[0]))
    current_time_vec = np.zeros((n_envs, N))  # array to save time
    while (episode_counts < episode_count_targets).any():
        observations += np.random.normal(0, 0.1, observations.shape)
        actions, states = model.predict(
            observations,  # type: ignore[arg-type]
            state=states,
            episode_start=episode_starts,
            deterministic=deterministic,
        )
        current_time_vec[:, timestep] = env.env_method("_get_time")
        current_episode_obs[:, timestep, :] = env.unnormalize_obs(observations)
        # current_episode_obs[:, timestep, :] = observations

        new_observations, rewards, dones, infos = env.step(actions)
        current_rewards += rewards
        current_lengths += 1

        current_episode_profits[:, timestep] = np.array([info["Profit"] for info in infos])
        current_episode_gains[:, timestep] = np.array([info["Gains"] for info in infos])
        current_episode_var_costs[:, timestep] = np.array([info["Variable costs"] for info in infos])
        current_episode_fixed_costs[:, timestep] = np.array([info["Fixed costs"] for info in infos])
        current_episode_actions[:, timestep, :] = np.array([info["Action"] for info in infos])
        # current_episode_cbuffer[:, timestep] = np.array([info["CBuffer"] for info in infos])
        # current_episode_fruit_weight[:, timestep] = np.array([info["Fruit Weight"] for info in infos])
        # current_episode_dli[:, timestep] = np.array([info["DLI"] for info in infos])

        timestep += 1

        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                # unpack values so that the callback can access the local variables
                # reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done


                if callback is not None:
                    callback(locals(), globals())

                if dones[i]:
                    if is_monitor_wrapped:
                        # Atari wrapper can send a "done" signal when
                        # the agent loses a life, but it does not correspond
                        # to the true end of episode
                        if "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            
                            episode_profits.append(current_episode_profits[i].copy())
                            episode_gains.append(current_episode_gains[i].copy())
                            episode_var_costs.append(current_episode_var_costs[i].copy())
                            episode_fixed_costs.append(current_episode_fixed_costs[i].copy())
                            episode_actions.append(current_episode_actions[i].copy())
                            episode_cbuffer.append(current_episode_cbuffer[i].copy())
                            episode_fruit_weight.append(current_episode_fruit_weight[i].copy())
                            episode_dli.append(current_episode_dli[i].copy())
                            episode_obs.append(current_episode_obs[i].copy())
                            time_vec.append(current_time_vec[i].copy())
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                            # set timestep for specific environement to 0
                            timestep = 0


                    else:
                        episode_rewards.append(current_rewards[i].copy())
                        episode_lengths.append(current_lengths[i].copy())
                        episode_profits.append(current_episode_profits[i].copy())
                        episode_gains.append(current_episode_gains[i].copy())
                        episode_var_costs.append(current_episode_var_costs[i].copy())
                        episode_fixed_costs.append(current_episode_fixed_costs[i].copy())
                        episode_actions.append(current_episode_actions[i].copy())
                        episode_cbuffer.append(current_episode_cbuffer[i].copy())
                        episode_fruit_weight.append(current_episode_fruit_weight[i].copy())
                        episode_dli.append(current_episode_dli[i].copy())
                        episode_obs.append(current_episode_obs[i].copy())
                        time_vec.append(current_time_vec[i].copy())

                        episode_counts[i] += 1
                        timestep = 0

                    current_rewards[i] = 0
                    current_lengths[i] = 0
                

        observations = new_observations
        if render:
            env.render()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    # if return_episode_rewards:
    return episode_rewards, episode_lengths, np.array(episode_actions), np.array(episode_obs), np.array(time_vec), np.array(episode_profits), np.array(episode_gains), np.array(episode_var_costs), np.array(episode_fixed_costs)
    # return mean_reward, std_reward, np.mean(episode_actions, axis=0), np.mean(episode_obs, axis=0), time_vec, episode_profits, episode_gains, episode_var_costs, episode_fixed_costs
