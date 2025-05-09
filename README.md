# GreenLight-Gym 2.0

## Reinforcement learning benchmark environment for control of greenhouse production systems 

<p align="center">
  <img src="./images/GLGymArchitecture2.svg" alt="GreenLight" width="700"/>
</p>


## Summary

**This repository is a reimplementation of the high-tech greenhouse model [GreenLight](https://github.com/davkat1/GreenLight) in `C++` with bindings for Python. The environment is designed to train reinforcement learning models to control greenhouse crop production systems.**


The code in this repository was used for the following [preprint](https://arxiv.org/abs/2410.05336) that has been accepted by [The 8th IFAC Conference on 
Sensing, Control and Automation Technologies for Agriculture](https://agricontrol25.sf.ucdavis.edu/).

✏ author: Bart van Laatum

📧 e-mail: bart.vanlaatum@wur.nl

## Installation

**Prerequisites**

Before installing and using the repository, make sure your system has the following:

- **C++ Compiler:**  
  A C++ compiler that supports C++17 (for example, GCC 7 or later, or Clang 6 or later).

- **Weights & Biases Account:**  
  A free account on [Weights & Biases](https://wandb.ai) is required to track experiments when using the provided RL scripts.

- **CasADi Library:**  
  Install the [CasADi](https://web.casadi.org/) library. On Linux you can download and install the pre-build CasADi libraries.

  Download the Linux Octave version from the release [website](https://web.casadi.org/get/). Make sure to grab `CasADi <= 3.5.5` since those have pre-built version (i.e., Binary Tarball). Extrat it somewhere convenient:
  
  ```shell
  sudo mkdir -p /opt/casadi-3.5.5
  sudo tar xzf casadi-linux-octave-6.1.0-v3.5.5.tar.gz -C /opt/casadi-3.5.5 --strip-components=1
  ```

  Move contents into `/usr/local` so that headers go to `/usr/local/include/casadi` and libs to `/usr/local/lib`:

  ```shell
  sudo cp -r /opt/casadi-3.7.0-linux64-octave7.3.0/include/* /usr/local/include/
  sudo cp    /opt/casadi-3.7.0-linux64-octave7.3.0/lib/libcasadi.so* /usr/local/lib/
  ```
  
  Refresh linker cache
  ```shell
  sudo ldconfig
  ```

  Additionally, you should ensure that the correct path is set:

  ```shell
  export CASADI_LIB=/path/to/casadi/lib
  export CPLUS_INCLUDE_PATH=/path/to/casadi/include
  ```

  Insert the right path where CasADi is installed `/path/to/`.

  > NOTE: Windows and macOS users installing CasADi is a bit more tricky. For Windows your best shot is using [`vcpkg`](https://vcpkg.io/en/). Please let me know whether you succeed the installation, such that we can add it to this installation guide.

___

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

## Future

We plan to extend GreenLight-Gym with the following features:

- **Python-native Model Implementation:**  
  Develop a pure Python version of the greenhouse model for easier maintenance, faster prototyping, and broader accessibility.

- **Model Predictive Control (MPC):**  
  Integrate MPC as an additional control baseline to benchmark against reinforcement learning algorithms.

- **Additional Crop Models:**  
  Add support for more crop types (e.g., cucumber, lettuce) to enable multi-crop benchmarking and research.

- **Adding more realistic energy systems:**  
  Precisely model the greenhouse energy consumption for heating, cooling, ventilation and lighting, via [EnergyPlus](https://energyplus.net/).

- **Improved Visualization Tools:**  
  Enhance the visualization suite for better analysis of experiments and model performance.


## Citing GreenLight-Gym

If you this repository and/or its accompanying article usefull, please cite it in your publications

```bibtex
@misc{vanlaatum2025greenlightgymreinforcementlearningbenchmark,
      title={GreenLight-Gym: Reinforcement learning benchmark environment for control of greenhouse production systems}, 
      author={Bart van Laatum and Eldert J. van Henten and Sjoerd Boersma},
      year={2025},
      eprint={2410.05336},
      archivePrefix={arXiv},
      primaryClass={eess.SY},
      url={https://arxiv.org/abs/2410.05336}, 
}
```
