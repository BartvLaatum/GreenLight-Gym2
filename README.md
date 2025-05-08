# GreenLight-Gym 2.0

## Reinforcement learning benchmark environment for control of greenhouse production systems 

![GreenLight](./images/GLGymArchitecture2.png)
___
### Summary

This repository is a reimplementation of the high-tech greenhouse model [GreenLight](https://github.com/davkat1/GreenLight) in `C++` with bindings for Python. The environment is desinged to train reinforcement learning models to control greenhouse crop production systems.


The code in this repository was used for the following [preprint](https://arxiv.org/abs/2410.05336) that has been accepted by [The 8th IFAC Conference on 
Sensing, Control and Automation Technologies for Agriculture](https://agricontrol25.sf.ucdavis.edu/).

✏ author: Bart van Laatum

📧 e-mail: bart.vanlaatum@wur.nl
___

### Installation

**Prerequisites**

Before installing and using the repository, make sure your system has the following:

- **C++ Compiler:**  
  A C++ compiler that supports C++17 (for example, GCC 7 or later, or Clang 6 or later).

- **CasADi Library:**  
  Install the [CasADi](https://web.casadi.org/) library. On Linux you can download and install the precompiled CasADi libraries. For example:
  
  ```shell
  wget https://web.casadi.org/files/casadi-py3-linux-64bit-release.tar.gz
  tar -xzvf casadi-py3-linux-64bit-release.tar.gz
  sudo cp libcasadi.so /usr/local/lib/
  sudo ldconfig
  ```

- **Weights & Biases Account:**  
  A free account on [Weights & Biases](https://wandb.ai) is required to track experiments when using the provided RL scripts.


1. **Clone the repository**
    ```shell
    git clone https://github.com/yourusername/GreenLight-Gym.git
    cd GreenLight-Gym
    ```

2. **Setup a Python virtual environment** 

    For instance, using anaconda

    ```shell
    conda create -n greenlight_gym python==3.11
    conda activate greenlight_gym
    ```

3. **Install the repository in Editable Mode**

   This repository is set up for an editable install using pip. From the root directory run:

   ```shell
   pip install -e .
   ```

   This command uses the setup.py file to build the C++ module (with dynamic paths) and install all Python packages. Adjust paths in `setup.py` if your libraries (like CasADi) are installed in different locations.


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

## Citing GreenLight-Gym

If you this repository and/or its accompanying article usefull, please cite it in your publications

```bibtex
@misc{vanlaatum2024greenlightgymreinforcementlearningbenchmark,
      title={GreenLight-Gym: Reinforcement learning benchmark environment for control of greenhouse production systems}, 
      author={Bart van Laatum and Eldert J. van Henten and Sjoerd Boersma},
      year={2025},
      eprint={2410.05336},
      archivePrefix={arXiv},
      primaryClass={eess.SY},
      url={https://arxiv.org/abs/2410.05336}, 
}
```
