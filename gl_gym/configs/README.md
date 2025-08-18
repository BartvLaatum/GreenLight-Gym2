## Configs overview

This directory contains YAML configuration files for:

- `envs/`: environment base and task-specific settings
- `agents/`: algorithm hyperparameters for training
- `sweeps/`: Weights & Biases sweep specs used for hyperparameter tuning

Below is a concise reference of keys and how they are used at runtime.

### Environment configs (`envs/`)

File: `envs/TomatoEnv.yml`

- `GreenLightEnv`
  - **weather_data_dir**: Root folder for weather CSVs (e.g., `gl_gym/environments/weather`).
  - **location**: Subfolder name under `weather_data_dir` (e.g., `Amsterdam`, `Spain`).
  - **num_params (int)**: Number of model parameters.
  - **nx (int)**: Number of states.
  - **nu (int)**: Number of control inputs.
  - **nd (int)**: Number of weather disturbances.
  - **dt (s)**: Solver step size (seconds).
  - **u_min / u_max (list<float>)**: Bounds for control inputs.
  - **delta_u_max (float)**: Max per-step change in control inputs.
  - **pred_horizon (days)**: Lookahead for weather predictions.
  - **season_length (days)**: Number of simulated days.
  - **start_train_year / end_train_year (int)**: Growth year range to sample from during training training CSVs.
  - **start_train_day / end_train_day (DOY)**: Start day of the year to sample during training.
  - **training (bool)**: Internal flag toggled by the runner; you normally don’t edit this.

- `TomatoEnv`
  - **reward_function**: Reward class name.
  - **reward_params**: Parameters for the reward function class (prices, penalty weights, etc.)
  - **observation_modules (list[str])**: Enabled observation modules.
  - **constraints**: Safety/comfort bounds
    - **co2_min/co2_max (ppm)**, **temp_min/temp_max (°C)**, **rh_min/rh_max (%)**
  - **eval_options**: Defaults for evaluation scripts
    - **eval_days (list<DOY>)**, **eval_years (list<int>)**, **location (str)**, **data_source (str)**

Notes
- Weather files must exist at `weather_data_dir/<location>/<year>.csv` with headers: `time,global radiation,wind speed,air temperature,sky temperature,??,CO2 concentration, day number,RH`.
- The environment resamples/interpolates weather to its solver step `dt`.

### Agent configs (`agents/`)

Agent YAMLs define hyperparameters per environment (top-level key `TomatoEnv`). At runtime, the training script reads:

- **n_envs** and **total_timesteps** to control vectorized envs and training length
- All other keys are forwarded to the Stable-Baselines3 model constructor (after light processing)

Common keys
- **policy**: Policy class string (e.g., `MlpPolicy`, `MlpLstmPolicy`).
- **learning_rate**: Float or schedule.
- **gamma**: Discount factor.
- **n_steps / batch_size / n_epochs**: Optimizer rollouts and batch settings.
- **clip_range** (PPO), **gae_lambda** (PPO), **vf_coef/ent_coef**, **max_grad_norm**
- **use_sde / sde_sample_freq**: State-dependent exploration options.
- **target_kl** (PPO): KL early stopping threshold or null.
- **policy_kwargs**: Nested settings forwarded to the policy
  - **net_arch**: `{pi: [..], vf: [..]}` or `{pi: [..], qf: [..]} (SAC)`
  - **optimizer_class**: `adam` or `rmsprop`
  - **optimizer_kwargs**: e.g., `{amsgrad: True}`
  - **activation_fn**: one of `silu`, `relu`, `tanh`, `elu`
  - **log_std_init**: Python expression string evaluated at runtime (e.g., `np.log(1)`)

SAC-specific
- **buffer_size**, **learning_starts**, **tau**, **train_freq**, **gradient_steps**
- **action_noise**: choose type and params, e.g.:
  - `action_noise.normalactionnoise.sigma: 0.05`
- **replay_buffer_class/replay_buffer_kwargs**, **optimize_memory_usage**, **target_update_interval**, **stats_window_size**

RecurrentPPO-specific
- **policy: MlpLstmPolicy**
- **policy_kwargs** adds:
  - **lstm_hidden_size (int)**, **n_lstm_layers (int)**, **shared_lstm (bool)**, **enable_critic_lstm (bool)**

### Sweep configs (`sweeps/`)

These are W&B Sweep specifications. The runner maps sampled configs to SB3 args and handles minor transforms:

General
- **method**: Sweep search strategy (e.g., `random`).
- **metric.name/goal**: Optimization target from training logs (e.g., `rollout/ep_rew_mean`).
- **parameters**: Grid or distributions for each tunable key.

Conventions used by the runner
- **gamma_offset**: The runner computes `gamma = 1.0 - gamma_offset` to sweep near 1.0.
- **activation_fn**: Mapped to torch classes: `silu`, `relu`, `tanh`, `elu`.
- **optimizer_kwargs**: Passed verbatim.
- SAC noise
  - **action_noise_type**: `normalactionnoise` or `ornstein_uhlenbeck`
  - **action_sigma**: Scalar sigma; runner builds the noise object with correct shape.

### How configs are used

- The training entrypoint loads env params from `envs/`, agent hyperparams from `agents/`, and — if `--hyperparameter_tuning` is enabled — creates a W&B sweep from `sweeps/` and uses sampled values.
- `n_envs` and `total_timesteps` are consumed by the runner; other agent keys are forwarded to SB3 after mapping `activation_fn`, `optimizer_class`, and optional SAC action noise.
- During evaluation, defaults in `TomatoEnv.eval_options` are used.


