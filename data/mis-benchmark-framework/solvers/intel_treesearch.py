import subprocess
import os.path
import pathlib
import shutil

import sys
import scipy.io
import numpy as np
import networkx as nx
import random as rd

from logzero import logger
from utils import launch_python_script_in_conda_env
from solvers.abstractsolver import MWISSolver

class IntelTreesearch(MWISSolver):
    def __init__(self) -> None:
        target_path = self.directory() / "NPHard"
        latest_kamis_hash = ""
        kamis_path = self.directory() / "KaMIS"

        if not target_path.exists():
            intel_repo = "https://github.com/isl-org/NPHard.git"
            target_commit = "5fc770ce1b1daee3cc9b318046f2361611894c27"
            subprocess.run(["git", "clone", intel_repo], cwd=self.directory())
            subprocess.run(["git","checkout",target_commit], cwd=target_path)
            subprocess.run(["git","apply", "../intel.patch"], cwd=target_path)

            if kamis_path.exists():
                shutil.rmtree(kamis_path)

        if not kamis_path.exists():
            kamis_repo = "https://github.com/KarlsruheMIS/KaMIS"
            target_commit = "791334ef6aebb2ccf005316258c043cca58923a0"
            subprocess.run(["git", "clone", kamis_repo], cwd=self.directory())
            subprocess.run(["git","checkout",target_commit], cwd=kamis_path)
            shutil.copytree(target_path / "kernel", kamis_path, dirs_exist_ok=True)
            subprocess.run(["make"], cwd=kamis_path)
            shutil.copyfile(kamis_path / "libreduce.so", target_path / "kernel/libreduce.so")

        self.model_state_path = None

    def load_weights(self, model_state_path):
        self.model_state_path = model_state_path

    def __str__(self) -> str:
        return "intel_treesearch"

    def directory(self):
        return pathlib.Path(__file__).parent / "intel_treesearch"

    def _prepare_instance(source_instance_file: pathlib.Path, cache_directory: pathlib.Path, **kwargs):
        cache_directory.mkdir(parents=True, exist_ok=True)

        dest_path = cache_directory / (source_instance_file.stem + ".mat")

        if os.path.exists(dest_path):
            source_mtime = os.path.getmtime(source_instance_file)
            last_updated = os.path.getmtime(dest_path)

            if source_mtime <= last_updated:
                return # we already have an up2date version of that file as matrix

        logger.info(f"Updated graph file: {source_instance_file}.")

        import dgl

        g = nx.read_gpickle(source_instance_file)
        labels_given = "label" in rd.sample(g.nodes(data=True), 1)[0][1].keys() # sample random node and check if we have a label
        
        # we use DGL to use some convience functions of bulk-converting attributes to int8
        g = dgl.from_networkx(g, node_attrs=["label"] if labels_given else [])

        indset_label = None
        if labels_given:
            g.ndata['label'] = g.ndata['label'].to(dgl.backend.data_type_dict['int8'])
            indset_label = np.expand_dims(g.ndata['label'].detach().numpy(), axis=1)

        adj = g.adjacency_matrix(False, scipy_fmt="csr")
        adj = adj.astype(np.float64)
        
        if labels_given:
            scipy.io.savemat(dest_path, { "adj": adj.tocsc(), "indset_label": indset_label })
        else:
            scipy.io.savemat(dest_path, { "adj": adj.tocsc() })

    def train(self, train_data_path: pathlib.Path, results_path: pathlib.Path, parameters):
        cache_directory = train_data_path / "preprocessed" / str(self)
        self._prepare_instances(train_data_path, cache_directory)

        logger.info("Invoking training of " + str(self))
        conda_env_name = str(self)
        conda_env_file = self.directory() / "environment.yml"
        script_file = self.directory() / "NPHard" / "train.py"

        arguments = [
            cache_directory, # input
            results_path # output
        ]

        if parameters["cuda_devices"]:
            arguments += ["--cuda_device", str(parameters["cuda_devices"][0])]

        if "model_prob_maps" in parameters.keys():
            arguments += ["--model_prob_maps", parameters["model_prob_maps"]]

        if "epochs" in parameters.keys():
            arguments += ["--epochs", parameters["epochs"]]

        if "lr" in parameters.keys():
            arguments += ["--lr", parameters["lr"]]

        if "weighted" in parameters.keys():
            logger.warning("weighted flag is not supported by Intel treesearch, ignoring...")

        if self.model_state_path:
            arguments += ["--pretrained_weights", self.model_state_path]

        if "self_loops" in parameters.keys():
            logger.warning("self loops enabled for Intel. This works in general, but all founds are of size 0! You probably do not want this.")
            arguments += ["--self_loops"]

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

        if parameters["num_threads"] == 1:
            script_file = self.directory() / "NPHard" / "demo.py"
        else:
            script_file = self.directory() / "NPHard" / "demo_parallel.py"

        arguments = [
            cache_directory, # input
            results_path, # output
            self.model_state_path,
            "--time_limit", parameters["time_limit"],
        ]

        if parameters["num_threads"] > 1:
            arguments += ["--num_threads", parameters["num_threads"]]

        if parameters["cuda_devices"]:
            arguments += ["--cuda_device", str(parameters["cuda_devices"][0])]

        if "model_prob_maps" in parameters.keys():
            arguments += ["--model_prob_maps", parameters["model_prob_maps"]]

        if "max_prob_maps" in parameters.keys():
            logger.warning("max_prob_maps flag is not supported by Intel treesearch, ignoring...")

        if "weighted" in parameters.keys():
            logger.warning("weighted flag is not supported by Intel treesearch, ignoring...")

        if "reduction" in parameters.keys():
            arguments += ["--reduction"]

        if "local_search" in parameters.keys():
            arguments += ["--local_search"]

        if "self_loops" in parameters.keys():
            logger.warning("self loops enabled for Intel. This works in general, but all founds are of size 0! You probably do not want this.")
            arguments += ["--self_loops"]

        logger.debug(f"Calling {script_file} with arguments {arguments}.")
        launch_python_script_in_conda_env(conda_env_name, conda_env_file, script_file, arguments)
