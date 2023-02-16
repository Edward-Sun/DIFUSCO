import subprocess
import os.path
import pathlib
import shutil

import scipy.io
import numpy as np
import networkx as nx
import random as rd

from logzero import logger
from utils import launch_python_script_in_conda_env
from solvers.abstractsolver import MWISSolver

class LearningWhatToDefer(MWISSolver):
    def __init__(self) -> None:
        target_path = self.directory() / "learning_what_to_defer"

        if not target_path.exists():
            intel_repo = "https://github.com/sungsoo-ahn/learning_what_to_defer"
            target_commit = "174ce85e6ab9dda0d1579caa4ebdb98ff60edcc5"
            subprocess.run(["git", "clone", intel_repo], cwd=self.directory())
            subprocess.run(["git","checkout",target_commit], cwd=target_path)
            subprocess.run(["git","apply", "../lwd.patch"], cwd=target_path)

        self.model_state_path = None

    def load_weights(self, model_state_path):
        self.model_state_path = model_state_path

    def __str__(self) -> str:
        return "learning_what_to_defer"

    def directory(self):
        return pathlib.Path(__file__).parent / "lwd"

    def _prepare_instance(source_instance_file: pathlib.Path, cache_directory: pathlib.Path, **kwargs):
        cache_directory.mkdir(parents=True, exist_ok=True)

        dest_path = cache_directory / (source_instance_file.stem + ".graph")

        if os.path.exists(dest_path):
            source_mtime = os.path.getmtime(source_instance_file)
            last_updated = os.path.getmtime(dest_path)

            if source_mtime <= last_updated:
                return # we already have an up2date version of that file as matrix

        logger.info(f"Updated graph file: {source_instance_file}.")

        g = nx.read_gpickle(source_instance_file)
        g.remove_edges_from(nx.selfloop_edges(g))
        
        # Write graph in gpickle format
        nx.write_gpickle(g, dest_path)


    def train(self, train_data_path: pathlib.Path, results_path: pathlib.Path, parameters):
        cache_directory = train_data_path / "preprocessed" / str(self)
        self._prepare_instances(train_data_path, cache_directory)

        logger.info("Invoking training of " + str(self))
        conda_env_name = str(self)
        conda_env_file = self.directory() / "environment.yml"
        script_file = self.directory() / "learning_what_to_defer" / "train_ppo.py"

        arguments = [
            "train",
            cache_directory, # input
            results_path, # output,
            "--num_samples", "2" # use 2 samples for training - taken from their code/paper
        ]

        if parameters["cuda_devices"]:
            arguments += ["--cuda_device", str(parameters["cuda_devices"][0])]

        if "weighted" in parameters.keys():
            logger.warning("weighted flag is not supported by LwD, ignoring...")

        if self.model_state_path:
            arguments += ["--pretrained_weights", self.model_state_path]

        if "maximum_iterations_per_episode" in parameters.keys():
            arguments += ["--maximum_iterations_per_episode", str(parameters["maximum_iterations_per_episode"])]

        if "num_unrolling_iterations" in parameters.keys():
            arguments += ["--num_unrolling_iterations", str(parameters["num_unrolling_iterations"])]

        if "num_environments_per_batch" in parameters.keys():
            arguments += ["--num_environments_per_batch", str(parameters["num_environments_per_batch"])]

        if "gradient_step_batch_size" in parameters.keys():
            arguments += ["--gradient_step_batch_size", str(parameters["gradient_step_batch_size"])]

        if "gradient_steps_per_update" in parameters.keys():
            arguments += ["--gradient_steps_per_update", str(parameters["gradient_steps_per_update"])]

        if "diversity_reward_coefficient" in parameters.keys():
            arguments += ["--diversity_reward_coefficient", str(parameters["diversity_reward_coefficient"])]

        if "max_entropy_coefficient" in parameters.keys():
            arguments += ["--max_entropy_coefficient", str(parameters["max_entropy_coefficient"])]

        if "num_updates" in parameters.keys():
            arguments += ["--num_updates", str(parameters["num_updates"])]

        if "training_graph_idx" in parameters.keys():
            arguments += ["--training_graph_idx", str(parameters["training_graph_idx"])]

        if "max_nodes" in parameters.keys():
            arguments += ["--max_nodes", str(parameters["max_nodes"])]

        logger.debug(f"Calling {script_file} with arguments {arguments}.")
        launch_python_script_in_conda_env(conda_env_name, conda_env_file, script_file, arguments)

    def solve(self, solve_data_path: pathlib.Path, results_path: pathlib.Path, parameters):
        if not self.model_state_path:
            raise ValueError("Cannot use solver without trained weights.")

        cache_directory = solve_data_path / "preprocessed" / str(self)
        self._prepare_instances(solve_data_path, cache_directory)

        logger.info("Solving all given instances using " + str(self))
        conda_env_name = str(self)
        conda_env_file = self.directory() / "environment.yml"
        script_file = self.directory() / "learning_what_to_defer" / "train_ppo.py"

        arguments = [
            "solve",
            cache_directory, # input
            results_path, # output
            "--pretrained_weights", self.model_state_path,
            "--num_samples", "10", # use 10 samples for solving - taken from their code/paper
            "--time_limit", parameters["time_limit"]
        ]

        if parameters["cuda_devices"]:
            arguments += ["--cuda_device", str(parameters["cuda_devices"][0])]

        if "maximum_iterations_per_episode" in parameters.keys():
            arguments += ["--maximum_iterations_per_episode", str(parameters["maximum_iterations_per_episode"])]

        if "max_nodes" in parameters.keys():
            arguments += ["--max_nodes", str(parameters["max_nodes"])]

        if "noise_as_prob_maps" in parameters.keys():
            arguments += ["--noise_as_prob_maps"]

        logger.debug(f"Calling {script_file} with arguments {arguments}.")
        launch_python_script_in_conda_env(conda_env_name, conda_env_file, script_file, arguments)
