# GreenLight-Gym 2.0

This repository is a reimplementation of the state-of-the-art greenhouse model in C++ with bindings for Python.

The model is wrapped in the Gymnasium environment to experiment (reinforcement) learning-based controllers.

Since the GreenLight model is implemented in CasADi, also *advanced* control techniques like MPC are available.

# Evaluate RL models

```shell
python experiments/evaluate_rl.py --project AgriControl --env_id TomatoEnv --model_name wise-elevator-233 --algorithm ppo
```

```shell
python experiments/evaluate_rl.py --project AgriControl --env_id TomatoEnv --model_name different-frog-235 --algorithm sac
```


Extract wandb information.

```shell
python processing/extract_wand_data.py --project AgriControl --group sac_det
```

Plotting the cost and state violation metrics 
```shell
python visualisations/cost_metrics.py --project AgriControl --mode deterministic --growth_year 2010 --start_day 59 --location Amsterdam
```

Plotting parametric uncertainties
```shell
python visualisations/param_uncertainty.py --project AgriControl --mode stochastic --growth_year 2010 --start_day 59 --location Amsterdam --algorithm ppo
```