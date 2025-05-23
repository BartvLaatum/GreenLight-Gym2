import os
import argparse
import gc
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# from sb3_contrib import RecurrentPPO
from torch.nn.modules.activation import ReLU, SiLU, Tanh, ELU
from torch.optim import Adam, RMSprop
from stable_baselines3 import PPO, SAC

from sb3_contrib import RecurrentPPO

from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

from gl_gym.common.utils import load_model_hyperparams
from gl_gym.RL.utils import (
    load_env_params, 
    wandb_init, 
    make_vec_env, 
    create_callbacks, 
    load_sweep_config
)

import wandb

ACTIVATION_FN = {"relu": ReLU, "silu": SiLU, "tanh":Tanh, "elu": ELU}
OPTIMIZER = {"adam": Adam, "rmsprop": RMSprop}
ACTION_NOISE = {"normalactionnoise": NormalActionNoise, "ornstein_uhlenbeck": OrnsteinUhlenbeckActionNoise}

def get_obs_names(env):
    """
    Extracts and returns a list of observation names from the given environment.

    Args:
        env: The environment object which contains observation modules.

    Returns:
        A list of strings where each string is an observation name with underscores replaced by spaces.
    """
    obs_names = []
    for obs_module in env.get_attr("observation_modules")[0]:
        for name in obs_module.obs_names:
            obs_names.append(name.replace("_", " "))
    return obs_names

class ExperimentManager:
    """Class to manage reinforcement learning experiments."""
    def __init__(
        self,
        env_id,
        project,
        env_base_params,
        env_specific_params,
        hyperparameters,
        group,
        n_eval_episodes,
        n_evals,
        algorithm,
        env_seed,
        model_seed,
        stochastic,
        save_model=True,
        save_env=True,
        hp_tuning=False,
        device="cpu"
    ):
        """
        Initialize the ExperimentManager with the given parameters.

        Args:
            env_id (str): ID of the environment to train on.
            project (str): Wandb project name.
            group (str): Wandb group name.
            total_timesteps (int): Total number of timesteps for training.
            n_eval_episodes (int): Number of episodes to evaluate the agent.
            num_cpus (int): Number of CPUs to use during training.
            n_evals (int): Number of evaluations during training.
            algorithm (str): RL algorithm to use.
            env_seed (int): Seed for the environment.
            model_seed (int): Seed for the model.
            save_model (bool): Whether to save the model.
            save_env (bool): Whether to save the environment.
            continue_training (bool): Whether to continue training from a saved model.
            continued_project (str): Project name of the saved model to continue training from.
            continued_runname (str): Run name of the saved model to continue training from.
        """
        self.env_id = env_id
        self.project = project
        self.env_base_params = env_base_params
        self.env_specific_params = env_specific_params
        self.n_envs = hyperparameters["n_envs"]
        self.total_timesteps = hyperparameters["total_timesteps"]
        self.stochastic = stochastic
        del hyperparameters["total_timesteps"] 
        del hyperparameters["n_envs"]
        self.hyperparameters = hyperparameters
        self.n_eval_episodes = n_eval_episodes
        self.group = group
        self.n_evals = n_evals
        self.algorithm = algorithm
        self.env_seed = env_seed
        self.model_seed = model_seed
        self.save_model = save_model
        self.save_env = save_env
        self.device = device
        # self.continue_training = continue_training
        # self.continued_project = continued_project
        # self.continued_runname = continued_runname
        self.hp_tuning = hp_tuning
        self.models = {"ppo": PPO, "sac": SAC, "recurrentppo": RecurrentPPO}

        self.model_class = self.models[self.algorithm.lower()]

        # Load environment and model parameters
        self.hyp_config_path = f"gl_gym/configs/sweeps/"


        # Initialize the environments
        print("Tuning:", self.hp_tuning)
        if not self.hp_tuning:
            self.run, self.config = wandb_init(
                self.hyperparameters,
                self.env_seed,
                self.model_seed,
                project=self.project,
                group=self.group,
                save_code=False
            )
            self.init_envs(self.hyperparameters["gamma"])
            self.model_params = self.build_model_parameters()

            # Initialize the model
            self.initialise_model()

    def init_envs(self, gamma):
        """
        Initialize training and evaluation environments
        """
        self.monitor_filename = None
        vec_norm_kwargs = {
            "norm_obs": True,
            "norm_reward": True,
            "clip_obs": 10,
            "gamma": gamma
        }

        # Setup new environment for training
        self.env_base_params["training"] = True
        self.env = make_vec_env(
            self.env_id,
            self.env_base_params,
            self.env_specific_params,
            seed=self.env_seed,
            n_envs=self.n_envs,  # Number of environments to run in parallel
            monitor_filename=self.monitor_filename,
            vec_norm_kwargs=vec_norm_kwargs
        )

        self.env_base_params["training"] = False
        self.eval_env = make_vec_env(
            self.env_id,
            self.env_base_params,
            self.env_specific_params,
            seed=self.env_seed,
            n_envs=1,                           # Only one environment for evaluation at the moment
            monitor_filename=self.monitor_filename,
            vec_norm_kwargs=vec_norm_kwargs,
            eval_env=True,
        )


    def initialise_model(self):
        """
        Initialize the model for training or continued training.

        Args:
            runname (str): Name of the run.
        """
        if self.stochastic:
            tensorboard_log = f"train_data/{self.project}/{self.algorithm}/stochastic/logs/{self.run.name}"
        else:
            tensorboard_log = f"train_data/{self.project}/{self.algorithm}/deterministic/logs/{self.run.name}"

        # Initialize a new model for training
        self.model = self.model_class(
            env=self.env,
            seed=self.model_seed,
            verbose=1,
            **self.model_params,
            tensorboard_log=tensorboard_log,
            device=self.device
        )

    def build_model_parameters(self):
        """
        Constructs and returns the model parameters for the reinforcement learning model.

        This method processes the hyperparameters provided during initialization to 
        configure the model parameters. It handles the conversion of string identifiers 
        for activation functions and optimizers into their respective classes. Additionally, 
        it sets up action noise parameters if the SAC algorithm is used.

        Returns:
            dict: A dictionary containing the model parameters, including any necessary 
                  conversions for activation functions, optimizers, and action noise.
        """
        model_params = self.hyperparameters.copy()

        if "policy_kwargs" in self.hyperparameters:
            policy_kwargs = self.hyperparameters["policy_kwargs"].copy()  # Copy to avoid modifying the original
            policy_kwargs["log_std_init"] = eval(policy_kwargs["log_std_init"])
            if "activation_fn" in policy_kwargs:
                activation_fn_str = policy_kwargs["activation_fn"]
                if activation_fn_str in ACTIVATION_FN:
                    policy_kwargs["activation_fn"] = ACTIVATION_FN[activation_fn_str]
                else:
                    raise ValueError(f"Unsupported activation function: {activation_fn_str}")

            # Handle other necessary conversions (e.g., optimizers)
            if "optimizer_class" in policy_kwargs:
                optimizer_str = policy_kwargs["optimizer_class"]
                if optimizer_str in OPTIMIZER:
                    policy_kwargs["optimizer_class"] = OPTIMIZER[optimizer_str]
                else:
                    raise ValueError(f"Unsupported optimizer: {optimizer_str}")
            model_params["policy_kwargs"] = policy_kwargs

        if self.algorithm == "sac":
            if "action_noise" in self.hyperparameters:
                action_noise_key, noise_params = next(iter(self.hyperparameters["action_noise"].items()))

                if action_noise_key in ACTION_NOISE:
                    action_noise = ACTION_NOISE[action_noise_key](
                        mean=np.zeros(self.env.action_space.shape),
                        sigma=noise_params["sigma"] * np.ones(self.env.action_space.shape)
                    )
                model_params["action_noise"] = action_noise
        return model_params


    def build_model_hyperparameters(self, config):
        """
        Builds RL model hyperparameters from a configuration file for Hyperparameter Tuning.
        The config file is generated by sampling from hyperparameter sweep.
        """
        self.model_params = dict(config).copy()
        self.model_params["gamma"] = 1.0 - config["gamma_offset"]

        policy_kwargs = {}
        policy_kwargs["net_arch"] = {}
        policy_kwargs["optimizer_kwargs"] = config["optimizer_kwargs"]
        policy_kwargs["activation_fn"] = ACTIVATION_FN[config["activation_fn"]]
        del self.model_params["optimizer_kwargs"], self.model_params["activation_fn"], self.model_params["gamma_offset"]

        if self.algorithm == "ppo":
            policy_kwargs["net_arch"]["pi"] = [config["pi"]]*3
            policy_kwargs["net_arch"]["vf"] = [config["vf"]]*3
            del self.model_params["vf"]

        elif self.algorithm == "sac":
            policy_kwargs["net_arch"]["pi"] = [config["pi"]]*3
            policy_kwargs["net_arch"]["qf"] = [config["qf"]]*3

            action_noise_key = config["action_noise_type"]
            action_std = config["action_sigma"]
            action_noise = ACTION_NOISE[action_noise_key](
                        mean=np.zeros(self.env.action_space.shape),
                        sigma=action_std * np.ones(self.env.action_space.shape)
                    )
            self.model_params["action_noise"] = action_noise
            del self.model_params["action_noise_type"],self.model_params["action_sigma"], self.model_params["qf"]

        elif self.algorithm == "recurrentppo":
            policy_kwargs["net_arch"]["pi"] = [config["pi"]]*2
            policy_kwargs["net_arch"]["vf"] = [config["vf"]]*2
            policy_kwargs["lstm_hidden_size"] = config["lstm_hidden_size"]
            policy_kwargs["enable_critic_lstm"] = config["enable_critic_lstm"]
            del self.model_params["vf"]

            if policy_kwargs["enable_critic_lstm"]:
                policy_kwargs["shared_lstm"] = False
            else:
                policy_kwargs["shared_lstm"] = True
            del self.model_params["lstm_hidden_size"], self.model_params["enable_critic_lstm"]

        del self.model_params["pi"]

        self.model_params["policy_kwargs"] = policy_kwargs

    def run_single_sweep(self):
        """
        Main function for hyperparameter tuning.
        """
        with wandb.init(sync_tensorboard=True) as run:
            self.run = run
            self.config = wandb.config
            self.init_envs(1-self.config["gamma_offset"])
            self.build_model_hyperparameters(self.config)
            self.initialise_model()
            self.run_experiment()

    def hyperparameter_tuning(self):
        """
        Perform hyperparameter tuning for the model. Using the Sweep API from Weights and Biases.
        """
        self.total_timesteps = 1.5e6
        continue_sweep = False
        sweep_config = load_sweep_config(self.hyp_config_path, self.env_id, self.algorithm)
        if continue_sweep:
            wandb.agent("puk5fznz", project="dwarf-env", function=self.run_single_sweep, count=100)
        else:
            sweep_id = wandb.sweep(sweep=sweep_config, project=self.project)
            wandb.agent(sweep_id, function=self.run_single_sweep, count=100)

    def run_experiment(self):
        """
        Executes the reinforcement learning experiment.

        This method manages the training process of the RL model using the initialized
        environments and model parameters. It sets up logging directories based on the 
        experiment"s stochastic or deterministic nature, calculates evaluation frequency, 
        and creates necessary callbacks for model evaluation and environment saving. 
        The method then trains the model for the specified number of timesteps, saves 
        the final model and environment normalization if required, and performs cleanup 
        operations post-training.
        """
        if self.stochastic:
            model_log_dir = f"train_data/{self.project}/{self.algorithm}/stochastic/models/{self.run.name}/" if self.save_model else None
            env_log_dir = f"train_data/{self.project}/{self.algorithm}/stochastic/envs/{self.run.name}/" if self.save_env else None
        else:
            model_log_dir = f"train_data/{self.project}/{self.algorithm}/deterministic/models/{self.run.name}/" if self.save_model else None
            env_log_dir = f"train_data/{self.project}/{self.algorithm}/deterministic/envs/{self.run.name}/" if self.save_env else None

        eval_freq = self.total_timesteps // self.n_evals // self.n_envs
        save_name = "vec_norm"

        callbacks = create_callbacks(
            self.n_eval_episodes,
            eval_freq,
            env_log_dir,
            save_name,
            model_log_dir,
            self.eval_env,
            run=self.run,
            results=None,
            save_env=self.save_env,
            verbose=1 # verbose-2; debug messages.
        )

        # Train the model
        self.model.learn(total_timesteps=self.total_timesteps, callback=callbacks, reset_num_timesteps=False)
        if model_log_dir:
            self.model.save(os.path.join(model_log_dir, "last_model"))

        # Save the environment normalization
        if env_log_dir:
            env_save_path = os.path.join(env_log_dir, "last_vecnormalize.pkl")
        self.model.get_vec_normalize_env().save(env_save_path)

        # Clean up and finalize the run
        self.run.finish()
        self.env.close()
        self.eval_env.close()
        del self.model, self.env, self.eval_env
        gc.collect()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="AgriControl", help="Wandb project name")
    parser.add_argument("--env_id", type=str, default="TomatoEnv", help="Environment ID")
    parser.add_argument("--algorithm", type=str, default="ppo", help="RL algorithm to use")
    parser.add_argument("--group", type=str, default="group1", help="Wandb group name")
    parser.add_argument("--n_eval_episodes", type=int, default=1, help="Number of episodes to evaluate the agent for")
    parser.add_argument("--n_evals", type=int, default=10, help="Number times we evaluate algorithm during training")
    parser.add_argument("--env_seed", type=int, default=666, help="Random seed for the environment for reproducibility")
    parser.add_argument("--model_seed", type=int, default=666, help="Random seed for the RL-model for reproducibility")
    parser.add_argument("--stochastic", action="store_true", help="Whether to run the experiment in stochastic mode")
    parser.add_argument("--device", type=str, default="cpu", help="The device to run the experiment on")
    parser.add_argument("--save_model", default=True, action=argparse.BooleanOptionalAction, help="Whether to save the model")
    parser.add_argument("--save_env", default=True, action=argparse.BooleanOptionalAction, help="Whether to save the environment")
    parser.add_argument("--hyperparameter_tuning", default=False, action=argparse.BooleanOptionalAction, help="Perform hyperparameter tuning")
    # parser.add_argument("--continue_training", default=False, action=argparse.BooleanOptionalAction, help="Continue training from a saved model")
    # parser.add_argument("--continued_project", type=str, default=None, help="Project name of the saved model to continue training from")
    # parser.add_argument("--continued_runname", type=str, default=None, help="Runname of the saved model to continue training from")
    args = parser.parse_args()

    env_config_path = f"gl_gym/configs/envs/"
    env_base_params, env_specific_params = load_env_params(args.env_id, env_config_path)
    hyperparameters = load_model_hyperparams(args.algorithm, args.env_id)
    # Initialize the experiment manager
    experiment_manager = ExperimentManager(
        env_id=args.env_id,
        project=args.project,
        env_base_params=env_base_params,
        env_specific_params=env_specific_params,
        hyperparameters=hyperparameters,
        group=args.group,
        n_eval_episodes=args.n_eval_episodes,
        n_evals=args.n_evals,
        algorithm=args.algorithm,
        env_seed=args.env_seed,
        model_seed=args.model_seed,
        stochastic=args.stochastic,
        save_model=args.save_model,
        save_env=args.save_env,
        hp_tuning=args.hyperparameter_tuning,
        device=args.device
    )

    if args.hyperparameter_tuning:
        # Perform hyperparameter tuning
        experiment_manager.hyperparameter_tuning()
    else:
    # Run the experiment
        experiment_manager.run_experiment()

if __name__ == "__main__":
    main()
