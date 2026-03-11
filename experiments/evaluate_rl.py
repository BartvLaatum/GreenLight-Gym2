"""
Evaluate a trained RL controller on GreenLightEnv.

Supports CLI overrides for:
  - weather scenarios   (--scenarios JSON)
  - parameter providers (--parameter_provider, --parameter_provider_kwargs)
  - reward / prices     (--reward_kwargs)

Example:
    python experiments/evaluate_rl.py \
        --algorithm ppo \
        --model_path train_data/AgriControl/ppo/deterministic/models/cosmic-music-45/best_model.zip \
        --vecnorm_path train_data/AgriControl/ppo/deterministic/envs/cosmic-music-45/best_vecnormalize.pkl \
        --scenarios '[{"location":"Amsterdam","growth_year":2010,"start_day":59}]' \
        --n_sims 5 \
        --save_dir data/rl/
"""
import argparse
import json
import os
from os.path import join

import numpy as np
import pandas as pd
from tqdm import tqdm

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, VecMonitor
from gymnasium.wrappers import FlattenObservation

from gl_gym.environments.greenlight_env import GreenLightEnv
from RL.utils import load_env_params, build_env_kwargs

ALGORITHMS = {"ppo": PPO, "sac": SAC}


def _make_env_factory(env_kwargs: dict, seed: int):
    """Return a zero-arg callable that builds a single GreenLightEnv."""
    def _init():
        env = GreenLightEnv(**env_kwargs)
        env = FlattenObservation(env)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        return env
    return _init


def build_vec_env(env_kwargs: dict, seed: int, vecnorm_path: str | None = None):
    """Create a DummyVecEnv, optionally wrapped in loaded VecNormalize."""
    vec_env = DummyVecEnv([_make_env_factory(env_kwargs, seed)])
    vec_env = VecMonitor(vec_env)

    if vecnorm_path is not None:
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    return vec_env


def evaluate_episode(model, vec_env, N: int) -> list[dict]:
    """Run one full episode and return per-step info dicts."""
    steps: list[dict] = []
    observations = vec_env.reset()
    states = None
    episode_starts = np.ones((1,), dtype=bool)

    for t in range(N):
        actions, states = model.predict(
            observations,
            state=states,
            episode_start=episode_starts,
            deterministic=True,
        )
        observations, rewards, dones, infos = vec_env.step(actions)
        info = infos[0]
        steps.append({
            "timestep": t,
            "reward": float(rewards[0]),
            **{k: v for k, v in info.items() if k != "controls"},
        })
        episode_starts = dones
        if dones[0]:
            break

    return steps


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained RL agent on the GreenLight environment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--env_id", default="GreenLightEnv")
    parser.add_argument("--env_config", default="gl_gym/configs/envs/",
                        help="Directory containing <env_id>.yml")
    parser.add_argument("--algorithm", default="ppo", choices=list(ALGORITHMS.keys()))
    parser.add_argument("--model_name", required=True,
                        help="Name of the saved model run (.zip)")
    parser.add_argument("--load_path", required=True,
                        help="Path to the saved models and environment folders")

    # --- scenario / domain-randomisation overrides ---
    parser.add_argument("--scenarios", type=str, default=None,
                        help='JSON list, e.g. \'[{"location":"Amsterdam","growth_year":2010,"start_day":59}]\'')
    parser.add_argument("--n_sims", type=int, default=1,
                        help="Repetitions per scenario (different seeds)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--parameter_provider", type=str, default=None,
                        help="Override parameter provider (fixed / randomized / set)")
    parser.add_argument("--parameter_provider_kwargs", type=str, default=None,
                        help="JSON dict of kwargs for the parameter provider")

    # --- reward / price overrides ---
    parser.add_argument("--reward_kwargs", type=str, default=None,
                        help='JSON dict, e.g. \'{"fruit_price": 2.0, "elec_price": 0.15}\'')

    parser.add_argument("--save_dir", default="results/rl/")
    args = parser.parse_args()

    # ---- load configs ----
    env_kwargs = load_env_params(args.env_id, args.env_config)
    env_kwargs, default_scenarios = build_env_kwargs(env_kwargs)

    # ---- apply CLI overrides ----
    scenarios = default_scenarios
    if args.scenarios is not None:
        scenarios = json.loads(args.scenarios)

    if args.parameter_provider is not None:
        env_kwargs["parameter_provider"] = args.parameter_provider
    if args.parameter_provider_kwargs is not None:
        env_kwargs["parameter_provider_kwargs"] = json.loads(args.parameter_provider_kwargs)

    if args.reward_kwargs is not None:
        overrides = json.loads(args.reward_kwargs)
        env_kwargs.setdefault("reward_kwargs", {}).update(overrides)

    # ---- load model (once) ----
    model = ALGORITHMS[args.algorithm].load(
        join(args.load_path, "models", args.model_name, "best_model"), device="cpu"
    )
    vec_norm_path = join(args.load_path, "envs", args.model_name, "best_vecnormalize.pkl")
    # ---- run evaluation per scenario ----
    os.makedirs(args.save_dir, exist_ok=True)
    all_results: list[dict] = []

    for scenario in scenarios:
        desc = f"{scenario['location']}/{scenario['growth_year']}d{scenario['start_day']}"

        # Pin the weather sampler to this scenario
        scenario_kwargs = env_kwargs.copy()
        scenario_kwargs["weather_scenario_sampler"] = "fixed"
        scenario_kwargs["weather_scenario_sampler_kwargs"] = {
            "location": scenario["location"],
            "growth_year": int(scenario["growth_year"]),
            "start_day": int(scenario["start_day"]),
        }

        for sim in tqdm(range(args.n_sims), desc=desc):
            vec_env = build_vec_env(
                scenario_kwargs,
                seed=args.seed + sim,
                vecnorm_path=vec_norm_path,
            )
            N = vec_env.get_attr("N")[0]
            episode_steps = evaluate_episode(model, vec_env, N)
            vec_env.close()

            for step in episode_steps:
                step.update(sim=sim, **scenario)
            all_results.extend(episode_steps)

    df = pd.DataFrame(all_results)
    save_path = join(args.save_dir, f"rl_{args.algorithm}_{args.env_id}.csv")
    df.to_csv(save_path, index=False)
    print(f"Saved {len(df)} rows to {save_path}")


if __name__ == "__main__":
    main()
