import copy
import os.path
import pathlib

import networkx as nx

from logzero import logger
from solvers.abstractsolver import MWISSolver
from utils import launch_python_script_in_conda_env

class Gurobi(MWISSolver):
    def __init__(self) -> None:
        pass

    def load_weights(self, model_state_path):
        raise NotImplementedError("Gurobi requires no weights.")

    def __str__(self) -> str:
        return "gurobi"

    def directory(self):
        return pathlib.Path(__file__).parent / "gurobi"

    @staticmethod
    def __prepare_graph(g: nx.Graph(), weighted = False):
        graph = copy.deepcopy(g)
        graph.remove_edges_from(nx.selfloop_edges(graph))
        # the gurobi solver file always expects a weighted file
        # however, for "unweighted" we supply all weights as 1
        if not weighted:
            nx.set_node_attributes(graph, 1, name="weight")

        return graph

    def _prepare_instance(source_instance_file: pathlib.Path, cache_directory: pathlib.Path, **kwargs):
        weighted = kwargs.get("weighted", False)
        cache_directory.mkdir(parents=True, exist_ok=True)

        dest_path = cache_directory / (source_instance_file.stem + f"_{'weighted' if weighted else 'unweighted'}.graph")
        
        if os.path.exists(dest_path):
            source_mtime = os.path.getmtime(source_instance_file)
            last_updated = os.path.getmtime(dest_path)

            if source_mtime <= last_updated:
                return # we already have an up2date version of that file

        logger.info(f"Updated graph file: {source_instance_file}.")
        
        g = nx.read_gpickle(source_instance_file)
        graph = Gurobi.__prepare_graph(g, weighted=weighted)
        nx.write_gpickle(graph, dest_path)

    def train(self, train_data_path: pathlib.Path, results_path: pathlib.Path, parameters):
        raise NotImplementedError("Gurobi cannot be trained!")

    def solve(self, solve_data_path: pathlib.Path, results_path: pathlib.Path, parameters):
        logger.info("Solving all given instances using " + str(self))

        weighted = "weighted" in parameters.keys()
        quadratic = "quadratic" in parameters.keys()
        write_mps = "write_mps" in parameters.keys()


        cache_directory = solve_data_path / "preprocessed" / str(self)
        self._prepare_instances(solve_data_path, cache_directory, weighted=weighted)

        conda_env_name = str(self)
        conda_env_file = self.directory() / "environment.yml"
        script_file = self.directory() / "main.py"

        arguments = [
            cache_directory, # input
            results_path, # output
            "--time_limit", parameters["time_limit"],
            "--loglevel", parameters["loglevel"],
            "--num_threads", parameters["num_threads"]
        ]

        if weighted:
            arguments += ["--weighted"]

        if quadratic:
            arguments += ["--quadratic"]

        if write_mps:
            arguments += ["--write_mps"]

        if "prm_file" in parameters.keys():
            arguments += ["--prm_file", parameters["prm_file"]]

        if weighted and quadratic:
            logger.error("Cannot use weighted and quadratic program together!")
            return

        logger.debug(f"Calling {script_file} with arguments {arguments}.")
        launch_python_script_in_conda_env(conda_env_name, conda_env_file, script_file, arguments)
