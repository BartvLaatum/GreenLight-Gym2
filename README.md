# GreenLight-Gym 2.0

## Reinforcement learning benchmark environment for control of greenhouse production systems 

<p align="center">
  <img src="./images/GLGymArchitecture2.svg" alt="GreenLight" width="700"/>
</p>


## Summary

**This repository is a reimplementation of the high-tech greenhouse model [GreenLight](https://github.com/davkat1/GreenLight) in the automatic differentiation tool `CasADi`. The environment is designed to train reinforcement learning models to control greenhouse crop production systems.**

The code in this repository was used for the following [preprint](https://arxiv.org/abs/2410.05336) that has been accepted by [The 8th IFAC Conference on 
Sensing, Control and Automation Technologies for Agriculture](https://agricontrol25.sf.ucdavis.edu/).

📄 preprint: https://arxiv.org/abs/2410.05336

✏ author: Bart van Laatum

📧 e-mail: bart.vanlaatum@wur.nl

### 📣 What changed in v0.2

Since we have moved from v0.1 to v0.2 this repository now fully relies on `Python`. Therefore, no complicated builds and pre-requirements are needed anymore. And one should be able to install and run this project on any platform (Windows, Ubuntu, MacOS). In summary:

> - **Python-native models**: no C++/CasADi build needed.
> - Cross-platform install via `pip`.
> - The previous C++ implementation is still available as **v0.1** (Ubuntu-focused).
>
> 🔗 For the C++/CasADi version, use tag **v0.1** (see Releases) and its README.

## Installation (v0.2, Python-native)

1. **Clone the repository**
    ```shell
    git clone https://github.com/BartvLaatum/GreenLight-Gym2.git
    cd GreenLight-Gym2
    ```

2. **Setup a Python virtual environment**

    For instance, using anaconda (or `python -m venv`)

    ```shell
    conda create -n greenlight_gym python==3.11
    conda activate greenlight_gym
    ```

3. **Install the repository in Editable Mode**

   This repository is set up for an editable install using pip. From the root directory run:

   ```shell
   pip install -e .
   ```

## Repository Structure

- The `gl_gym/` folder contains:
    - Environment code under [`environments/`](./gl_gym/environments) (models, dynamics, parameters, and utility functions).
    - Configuration files under [`configs/`](./gl_gym/configs).
    - Common utility functions under [`common/`](./gl_gym/common).
    - The [`RL/`](./gl_gym/RL) folder contains, the experiment manager (experiment_manager.py) that sets up training, evaluation, hyperparameter tuning (using Weights & Biases), etc.
    - The [`experiments/`](./gl_gym/experiments) folder contains: Experiment `Python` scripts (e.g. RL training or evaluation)
    - These experiments scripts can be called through bash scripts placed in [`run_scripts/`](./run_scripts). 

## Usage

1. **Running an RL Experiment**

To start a new reinforcement learning experiment run. One can change the environment, model, etc via the flags in that bash script.

In the configuration files one can change hyperparameters for [`PPO`](./gl_gym/configs/agents/ppo.yml) or the [`environment`](./gl_gym/configs/envs/TomatoEnv.yml) specific parameters.

```bash
bash run_scripts/rl.sh
```

2. **Evaluation of Trained Models**
You can evaluate trained models using the evaluation scripts provided in the experiments folder `evaluate_rl.py`:

```shell
python gl_gym/experiments/evaluate_rl.py --project PROJECT_NAME --env_id TomatoEnv --model_name YOUR_MODEL_NAME --algorithm ppo
```

3. **Evaluation of Baseline Controller**
You can evaluate the rule-based baseline controller for different levels of parametric uncertainty through the following bash script:

```bash
bash run_scripts/eval_baseline.sh
```


4. **Visualizations**
    - **Plotting**: The repository includes scripts under [visualisations](./visualisations/) for plotting learning curves and cost metrics. 
    - Before, generating any plots you must have evaluated your RL agents with `evaluate_rl.py` and a rule-based baseline with `evaluate_baseline.py`


    #### Time-series of trajectories.
    Compares the state and control input trajectories for $N$ consecutive days.
    ```shell
    python visualisations/trajectories.py --project PROJECT_NAME --MODE --ppo_name PPO_MODEL_NAME --sac_name SAC_MODEL_NAME --growth_year GROWTH_YEAR --start_day START_DAY --location LOCATION --n_days2plot NUMBER_OF_DAY_TO_VISUALIZE --uncertainty_value UNCERTAINTY_VALUE
    ```    
    <p align="center">
      <img src="./images/timeseries_state.png" alt="Time series state" width="400"/>
    </p>

    #### Bar plot the performance metrics.
    Creates a bar plot of controller performance regarding cost en constraints metrics.
    ```shell
    python visualisations/cost_metrics.py --project PROJECT_NAME --MODE --uncertainty_value UNCERTAINTY_VALUE --growth_year GROWTH_YEAR --start_day START_DAY --location LOCATION
    ```
    #### Example of comparing SAC, PPO and rule-based agent (RB) on economic performance indicator (EPI) metrics.
    <p align="center">
      <img src="./images/cost_metrics_comparison.png" alt="Cost Metrics Comparison" width="400"/>
    </p>

    #### Line plot the performance metrics over parametric uncertainty scale.
    Visualizes how the cumulative reward changes with different levels of parametric uncertainty in the environment by comparing controller performance. NOTE: Don't forget to update variable `model_names` to the correct model names in [`param_uncertainty.py`](./visualisations/param_uncertainty.py). 
    ```shell
    python visualisations/param_uncertainty.py --project PROJECT_NAME --mode MODE --growth_year GROWTH_YEAR --start_day START_DAY --location LOCATION
    ```
      #### Example of comparing SAC, PPO and rule-based agent (RB) on the cumulative reward trained per parametric uncertainty environment.
    <p align="center">
      <img src="./images/cumulative_reward.png" alt="Performance uncertainty" width="400"/>
    </p>

> Note that the other three scripts in `visualisations/` require additional data, which can be made available upon request.

## Future road map

We plan to extend GreenLight-Gym with the following features:

- [x] ~~**Python-native Model Implementation:**~~
  Develop a pure Python version of the greenhouse model for easier maintenance, faster prototyping, and broader accessibility. *(Implemented in v0.2)*

- **Model Predictive Control (MPC):**  
  Integrate MPC as an additional control baseline to benchmark against reinforcement learning algorithms.

- **Additional Crop Models:**  
  Add support for more crop types (e.g., cucumber, lettuce) to enable multi-crop benchmarking and research.

- **Adding more realistic energy systems:**  
  Precisely model the greenhouse energy consumption for heating, cooling, ventilation and lighting, via [EnergyPlus](https://energyplus.net/).

- **Improved Visualization Tools:**  
  Enhance the visualization suite for better analysis of experiments and model performance.


## Citation

If you find this repository and/or its accompanying article usefull, please cite it in your publications

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
