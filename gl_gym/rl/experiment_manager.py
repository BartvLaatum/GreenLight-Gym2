import copy
import os
import argparse
from multiprocessing import cpu_count
import gc
from copy import copy
from pprint import pprint
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from torch.nn.modules.activation import ReLU, SiLU, Tanh, ELU
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv, VecMonitor

from gl_gym.experiments.utils import load_env_params, load_model_params, wandb_init, make_vec_env, create_callbacks, make_env, load_sweep_config
from gl_gym.common.results import Results

import wandb

ACTIVATION_FN = {"ReLU": ReLU, "SiLU": SiLU, "Tanh":Tanh, "ELU": ELU}

def get_obs_names(env):
    """
    Extracts and returns a list of observation names from the given environment.

    Args:
        env: The environment object which contains observation modules.

    Returns:
        A list of strings where each string is an observation name with underscores replaced by spaces.
    """
    obs_names = []
    for obs_module in env.get_attr('observation_modules')[0]:
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
        result_columns,
        group,
        total_timesteps,
        n_eval_episodes,
        num_cpus,
        n_evals,
        algorithm,
        env_seed,
        model_seed,
        save_model=True,
        save_env=True,
        continue_training=False,
        continued_project=None,
        continued_runname=None,
        hp_tuning=False
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
        self.result_columns = result_columns
        self.group = group
        self.total_timesteps = total_timesteps
        self.n_eval_episodes = n_eval_episodes
        self.num_cpus = num_cpus
        self.n_evals = n_evals
        self.algorithm = algorithm
        self.env_seed = env_seed
        self.model_seed = model_seed
        self.save_model = save_model
        self.save_env = save_env
        self.continue_training = continue_training
        self.continued_project = continued_project
        self.continued_runname = continued_runname
        self.hp_tuning = hp_tuning
        self.models = {'ppo': PPO, 'recurrentppo': RecurrentPPO}

        self.model_class = self.models[self.algorithm.lower()]

        # Load environment and model parameters
        # self.env_config_path = f"gl_gym/configs/envs/"
        self.model_config_path = f"gl_gym/configs/agents/"
        self.hyp_config_path = f"gl_gym/configs/sweeps/"
        # self.env_base_params, self.env_specific_params, self.result_columns = load_env_params(self.env_id, self.env_config_path)

        self.model_params = load_model_params(self.algorithm, self.model_config_path, self.env_id)


        # Initialize the environments
        print("Tuning:", self.hp_tuning)
        if not self.hp_tuning:
            self.run, self.config = wandb_init(
                self.model_params,
                self.env_base_params,
                self.env_specific_params,
                self.total_timesteps,
                self.env_seed,
                self.model_seed,
                project=self.project,
                group=self.group,
                runname=None,
                job_type="train",
                save_code=True
            )

            self.init_envs()
            print(self.env.observation_space.shape)
            # Initialize the model
            self.initialise_model()

    def init_envs(self):
        '''
        Initialize training and evaluation environments
        '''
        self.monitor_filename = None
        vec_norm_kwargs = {"norm_obs": True, "norm_reward": True, "clip_obs": 50_000}

        if self.continue_training:
            print("Continuing training from a saved model")

            env_path = f"train_data/{self.continued_project}/envs/{self.continued_runname}"
            # # Setup environment for continued training
            self.env = make_vec_env(self.env_id,
                self.env_base_params,
                self.env_specific_params,
                seed=self.env_seed,
                n_envs= self.num_cpus,
                monitor_filename=self.monitor_filename,
                vec_norm_kwargs=None,
                eval_env=True)
            self.env = VecNormalize.load(os.path.join(env_path, "best_vecnormalize.pkl"), self.env)
            
            self.env_base_params["training"] = False
            self.eval_env = make_vec_env(self.env_id,
                self.env_base_params,
                self.env_specific_params,
                seed=self.env_seed,
                n_envs= 1,
                monitor_filename=self.monitor_filename,
                vec_norm_kwargs=None,
                eval_env=True)

            self.eval_env = VecNormalize.load(os.path.join(env_path, "best_vecnormalize.pkl"), self.eval_env)
        else:
            # Setup new environment for training
            self.env_base_params["training"] = True
            self.env = make_vec_env(
                self.env_id,
                self.env_base_params,
                self.env_specific_params,
                seed=self.env_seed,
                n_envs=self.num_cpus,
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
        obs_names = get_obs_names(self.env)
        cols = copy(self.result_columns)
        cols[1:1] = obs_names
        print(cols)
        print(len(cols))
        self.results = Results(cols)


    def initialise_model(self):
        '''
        Initialize the model for training or continued training.

        Args:
            runname (str): Name of the run.
            job_type (str): Type of job (default is 'train').
        '''

        tensorboard_log = f"train_data/{self.project}/logs/{self.run.name}"

        if self.continue_training:
            custom_objects = {'batch_size': 64, 'ent_coef': 0.01, 'n_epochs': 8, 'learning_rate': 2e-6, 'n_steps': 2048}
            self.model_params.update(custom_objects)
            pprint(self.model_params)
            wandb.config.update({'model_params': {**self.model_params}}, allow_val_change=True)
            print("loading saved model for continued training")
            tensorboard_log = f"train_data/{self.continued_project}/logs/{self.run.name}"

            # Load the model for continued training
            model_path = f"train_data/{self.continued_project}/models/{self.continued_runname}/best_model.zip"
            self.model = self.model_class.load(model_path, env=self.env, tensorboard_log=tensorboard_log, custom_objects=custom_objects)
            print(self.model.ent_coef)
        else:
            # Initialize a new model for training
            self.model = self.model_class(
                env=self.env,
                seed=self.model_seed,
                verbose=1,
                **self.model_params,
                tensorboard_log=tensorboard_log
            )

    def build_model_hyperparameters(self, config):
        """Build the model hyperparameters from the given config."""

        self.model_params["policy"] = config["policy"]
        self.model_params["learning_rate"] = config["learning_rate"]
        self.model_params["n_steps"] = config["n_steps"]
        self.model_params["batch_size"] = config["batch_size"]
        self.model_params["n_epochs"] = config["n_epochs"]
        self.model_params["gamma"] = 1.0 - config["gamma_offset"]
        self.model_params["gae_lambda"] = config["gae_lambda"]
        self.model_params["clip_range"] = config["clip_range"]
        self.model_params["ent_coef"] = config["ent_coef"]
        self.model_params["vf_coef"] = config["vf_coef"]

        self.model_params["use_sde"] = config["use_sde"]
        self.model_params["sde_sample_freq"] = config["sde_sample_freq"]
        self.model_params["target_kl"] = config["target_kl"]
        self.model_params["normalize_advantage"] = config["normalize_advantage"]

        policy_kwargs = {}
        policy_kwargs["net_arch"] = {}
        
        if self.algorithm == "ppo":
            policy_kwargs["net_arch"]["pi"] = [config["pi"]]*3
            policy_kwargs["net_arch"]["vf"] = [config["vf"]]*3

        policy_kwargs["optimizer_kwargs"] = config["optimizer_kwargs"]
        policy_kwargs["activation_fn"] = ACTIVATION_FN[config["activation_fn"]]

        if self.algorithm == "recurrentppo":
            policy_kwargs["net_arch"]["pi"] = [config["pi"]]*2
            policy_kwargs["net_arch"]["vf"] = [config["vf"]]*2
            policy_kwargs["lstm_hidden_size"] = config["lstm_hidden_size"]
            policy_kwargs["enable_critic_lstm"] = config["enable_critic_lstm"]
            if policy_kwargs["enable_critic_lstm"]:
                policy_kwargs["shared_lstm"] = False
            else:
                policy_kwargs["shared_lstm"] = True
        self.model_params["policy_kwargs"].update(policy_kwargs)

    def run_single_sweep(self):
        """
        Main function for hyperparameter tuning.
        """
        # wandb.tensorboard.patch(root_logdir="...")
        with wandb.init(sync_tensorboard=True) as run:
            self.run = run
            self.config = wandb.config
            self.build_model_hyperparameters(self.config)
            self.init_envs()
            self.initialise_model()
            print(self.model.policy) 
            self.run_experiment()

    def hyperparameter_tuning(self):
        '''
        Perform hyperparameter tuning for the model. Using the Sweep API from Weights and Biases.
        '''
        continue_sweep = True
        sweep_config = load_sweep_config(self.hyp_config_path, self.env_id, self.algorithm)
        if continue_sweep:
            wandb.agent("puk5fznz", project="dwarf-env", function=self.run_single_sweep, count=100)
        else:
            sweep_id = wandb.sweep(sweep=sweep_config, project=self.project)
            wandb.agent(sweep_id, function=self.run_single_sweep, count=100)

    def run_experiment(self):
        '''Run the experiment with the initialized model and environments.'''
        model_log_dir = f"train_data/{self.project}/models/{self.run.name}/" if self.save_model else None
        env_log_dir = f"train_data/{self.project}/envs/{self.run.name}/" if self.save_env else None

        eval_freq = self.total_timesteps // self.n_evals // self.num_cpus
        save_name = "vec_norm"

        callbacks = create_callbacks(
            self.n_eval_episodes,
            eval_freq,
            env_log_dir,
            save_name,
            model_log_dir,
            self.eval_env,
            run=self.run,
            results=self.results,
            save_env=self.save_env,
            verbose=0 # verbose-2; debug messages.
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="DwarfTomatoes", help="Environment ID")
    parser.add_argument("--project", type=str, default="casadi", help="Wandb project name")
    parser.add_argument("--group", type=str, default="group1", help="Wandb group name")
    parser.add_argument("--total_timesteps", type=int, default=50_000, help="Total number of timesteps to train algorithm for")
    parser.add_argument("--n_eval_episodes", type=int, default=1, help="Number of episodes to evaluate the agent for")
    parser.add_argument("--num_cpus", type=int, default=1, help="Number of CPUs to use during training")
    parser.add_argument("--n_evals", type=int, default=5, help="Number times we evaluate algorithm during training")
    parser.add_argument("--algorithm", type=str, default="ppo", help="RL algorithm to use")
    parser.add_argument("--env_seed", type=int, default=666, help="Random seed for the environment for reproducibility")
    parser.add_argument("--model_seed", type=int, default=666, help="Random seed for the RL-model for reproducibility")
    parser.add_argument('--save_model', default=True, action=argparse.BooleanOptionalAction, help="Whether to save the model")
    parser.add_argument("--save_env", default=True, action=argparse.BooleanOptionalAction, help="Whether to save the environment")
    parser.add_argument("--hyperparameter_tuning", default=False, action=argparse.BooleanOptionalAction, help="Perform hyperparameter tuning")
    parser.add_argument("--continue_training", default=False, action=argparse.BooleanOptionalAction, help="Continue training from a saved model")
    parser.add_argument("--continued_project", type=str, default=None, help="Project name of the saved model to continue training from")
    parser.add_argument("--continued_runname", type=str, default=None, help="Runname of the saved model to continue training from")
    args = parser.parse_args()

    assert args.num_cpus <= cpu_count(), f"Number of CPUs requested ({args.num_cpus}) is greater than available ({cpu_count()})"
    env_config_path = f"gl_gym/configs/envs/"
    env_base_params, env_specific_params, result_columns = load_env_params(args.env_id, env_config_path)

    # Initialize the experiment manager
    experiment_manager = ExperimentManager(
        env_id=args.env_id,
        project=args.project,
        env_base_params=env_base_params,
        env_specific_params=env_specific_params,
        result_columns=result_columns,
        group=args.group,
        total_timesteps=args.total_timesteps,
        n_eval_episodes=args.n_eval_episodes,
        num_cpus=args.num_cpus,
        n_evals=args.n_evals,
        algorithm=args.algorithm,
        env_seed=args.env_seed,
        model_seed=args.model_seed,
        save_model=args.save_model,
        save_env=args.save_env,
        continue_training=args.continue_training,
        continued_project=args.continued_project,
        continued_runname=args.continued_runname,
        hp_tuning=args.hyperparameter_tuning
    )

    if args.hyperparameter_tuning:
        # Perform hyperparameter tuning
        experiment_manager.hyperparameter_tuning()
    else:
    # Run the experiment
        experiment_manager.run_experiment()
