# GreenLight-Gym 2.0

## Reinforcement learning benchmark environment for control of greenhouse production systems 

### Summary

This repository is a reimplementation of the high-tech greenhouse model [GreenLight](https://github.com/davkat1/GreenLight) in `C++` with bindings for Python. The environment is desinged to train reinforcement learning models to control greenhouse crop production systems.


The code in this repository was used for the following [preprint](https://arxiv.org/abs/2410.05336) that has been accepted by [The 8th IFAC Conference on 
Sensing, Control and Automation Technologies for Agriculture](https://agricontrol25.sf.ucdavis.edu/).

✏ author: Bart van Laatum

📧 e-mail: bart.vanlaatum@wur.nl
___

### Installation

1. **Clone the repository**
```shell
git clone https://github.com/yourusername/GreenLight-Gym.git
cd GreenLight-Gym
```

2. **Setup a Python virtual environment** (for instance using anaconda)

```shell
conda create -n greenlight_gym python==3.11
```
___

### Repository Structure

- The `gl_gym/` folder contains:

    - Environment code under [`environments`](./gl_gym/environments) (models, dynamics, parameters, and utility functions).
    - Configuration files under [`configs`](./gl_gym/configs).
    - Common utility functions under [`common`](./gl_gym/common).
- The `experiments/` folder contains:

    - Experiment scripts (e.g. RL training or evaluation – see `experiments/rl.sh`).

- The `RL/` folder contains:
    - The experiment manager (RL/experiment_manager.py) that sets up training, evaluation, hyperparameter tuning (using Weights & Biases), etc.
___
### Usage

1. **Running an RL Experiment**

To start a new reinforcement learning experiment using (for example) PPO on the Tomato environment, run:

```shell
python RL/experiment_manager.py --env_id TomatoEnv --algorithm ppo
```

2. **Evaluation of Trained Models**
You can evaluate pre-trained models using the evaluation scripts provided in the experiments folder `evaluate_rl.py`:

```shell
python experiments/evaluate_rl.py --project PROJECT_NAME --env_id TomatoEnv --model_name YOUR_MODEL_NAME --algorithm ppo
```

3. **Visualizations**
    - **Plotting**: The repository includes scripts under [visualisations](./visualisations/) for plotting learning curves and cost metrics
___

### Notes

Adjust paths in `setup.py` if your libraries (like `CasADi`) are installed in different locations. The repository is designed as a reinforcement learning environment for greenhouse crop production. The environment ([TomatoEnv](./gl_gym/environments/tomato_env.py)) are configurable via the config files in envs. 
