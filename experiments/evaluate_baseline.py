"""
Evaluate the rule-based baseline controller on GreenLightEnv.

Supports CLI overrides for:
  - weather scenarios   (--scenarios JSON)
  - parameter providers (--parameter_provider, --parameter_provider_kwargs)
  - reward / prices     (--reward_kwargs)

Example:
    python experiments/evaluate_baseline.py \
        --scenarios '[{"location":"Amsterdam","growth_year":2010,"start_day":59}]' \
        --n_sims 1 \
        --save_dir data/baseline/
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from gl_gym.environments.greenlight_env import GreenLightEnv
from gl_gym.components.rule_based import RuleBasedController
from gl_gym.core.types import StepContext
from RL.utils import load_env_params, load_model_hyperparams, build_env_kwargs


def build_step_context(env: GreenLightEnv) -> StepContext:
    return StepContext(
        t=env.timestep, dt=env.dt, Np=env.Np,
        x_prev=env.x_prev, x=env.x, u=env.u, p=env.p,
        d=env.weather_data,
        hour_of_day=env.hour_of_day, day_of_year=env.day_of_year,
    )


def evaluate_episode(env: GreenLightEnv, controller: RuleBasedController) -> list[dict]:
    """Run one full episode and return per-step info dicts."""
    steps: list[dict] = []
    while True:
        ctx = build_step_context(env)
        u = controller.predict(ctx)
        _obs, reward, terminated, truncated, info = env.step(u.astype(np.float32))
        steps.append({
            "timestep": env.timestep,
            "reward": float(reward),
            **{k: v for k, v in info.items() if k != "controls"},
        })
        if terminated or truncated:
            break
    return steps


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the rule-based baseline on the GreenLight environment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--env_id", default="GreenLightEnv")
    parser.add_argument("--env_config", default="gl_gym/configs/envs/",
                        help="Directory containing <env_id>.yml")
    parser.add_argument("--rb_config", default="configs/agents/",
                        help="Directory containing rule_based.yml")

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

    parser.add_argument("--save_dir", default="results/baseline/")
    args = parser.parse_args()

    # ---- load configs ----
    env_kwargs = load_env_params(args.env_id, args.env_config)
    env_kwargs, default_scenarios = build_env_kwargs(env_kwargs)
    rb_params = load_model_hyperparams("rule_based", args.env_id)

    # Bypass action-scheme normalisation so raw controls pass through
    env_kwargs["normalize_actions"] = False

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

    # ---- build env + controller ----
    env = GreenLightEnv(**env_kwargs)
    controller = RuleBasedController(**rb_params)

    # ---- run evaluation ----
    os.makedirs(args.save_dir, exist_ok=True)
    all_results: list[dict] = []

    for scenario in scenarios:
        desc = f"{scenario['location']}/{scenario['growth_year']}d{scenario['start_day']}"
        for sim in tqdm(range(args.n_sims), desc=desc):
            env.reset(seed=args.seed + sim, options={"scenario": scenario})
            episode_steps = evaluate_episode(env, controller)
            for step in episode_steps:
                step.update(sim=sim, **scenario)
            all_results.extend(episode_steps)

    df = pd.DataFrame(all_results)
    save_path = os.path.join(args.save_dir, f"rb_baseline_{args.env_id}.csv")
    df.to_csv(save_path, index=False)
    print(f"Saved {len(df)} rows to {save_path}")


if __name__ == "__main__":
    main()
