import os
import argparse

import wandb
import numpy as np
import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", type=str, default="bartvlaatum", help="W&B entity name, usually your username")
    parser.add_argument("--project", type=str, help="name of the project from which you want to extract logged data")
    parser.add_argument("--group", type=str, help="name of the group from which you want to extract logged data")
    args = parser.parse_args()

    save_dir = f"data/{args.project}/deterministic/{args.group}/"
    os.makedirs(save_dir, exist_ok=True)

    api = wandb.Api()
    runs = api.runs(args.entity + "/" + args.project, filters={"group": args.group})
    print(runs)
    summary_list, config_list, name_list = [], [], []

    train_rew_mean = []
    step = []
    global_step = []

    for run in runs:
        # .summary contains output keys/values for
        # metrics such as accuracy.
        #  We call ._json_dict to omit large files
        summary_list.append(run.summary._json_dict)
        summary_dict = run.summary._json_dict
        history_dict = run.history()

        train_rew_mean.append(history_dict['rollout/ep_rew_mean'])

        N = len(history_dict['rollout/ep_rew_mean'])
        
        step.append(np.arange(1, N+1))
        name_list.append([run.name]*N)
        global_step.append(history_dict['global_step'])

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})

    step = np.array(step[1:]).flatten()
    train_rew_mean = np.array(train_rew_mean[1:]).flatten()
    name_list = np.array(name_list[1:]).flatten()
    global_step = np.array(global_step[1:]).flatten()

    print(len(step), len(train_rew_mean), len(name_list), len(global_step))

    train_df = pd.DataFrame({'global step': global_step,"step": step, "train reward": train_rew_mean, "name": name_list})

    train_df = train_df.dropna()
    train_df.to_csv(f"data/{args.project}/deterministic/{args.group}/rollout.csv", index=False)

# extract_wandb_data()
