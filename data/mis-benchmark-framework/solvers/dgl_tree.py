from solvers.abstractsolver import MWISSolver
import os.path
import pathlib
import dgl
import networkx as nx
import numpy as np
import random as rd
import json
from tqdm import tqdm
from utils import launch_python_script_in_conda_env

from logzero import logger

class DGLTreesearch(MWISSolver):
    def __init__(self):
        self.model_state_path = None

    def load_weights(self, model_state_path):
        self.model_state_path = model_state_path

    def __str__(self):
        return "dgl_treesearch"

    def directory(self):
        return pathlib.Path(__file__).parent / "dgl_treesearch"

    def _prepare_instance(source_instance_file, cache_directory, **kwargs):
        raise NotImplementedError("This function should never be called.")

    @staticmethod
    def __prepare_graph(nx_graph, weighted=False):
        labels_given = "label" in rd.sample(nx_graph.nodes(data=True), 1)[0][1].keys() # sample random node and check if we have a label
        node_attrs = []

        if labels_given:
            node_attrs.append("label")

        if weighted:
            node_attrs.append("weight")

        g = dgl.from_networkx(nx_graph, node_attrs=node_attrs)
        
        if labels_given:
            g.ndata['label'] = g.ndata['label'].to(dgl.backend.data_type_dict['int8'])

        if not weighted:
            g.ndata['weight'] = dgl.backend.tensor(np.ones(shape=(g.num_nodes(),1)), dtype=dgl.backend.data_type_dict['float16'])
        
        # force shape of weights (n x 1 shape)
        g.ndata['weight'] = g.ndata['weight'].reshape(-1, 1).to(dgl.backend.data_type_dict['float16'])

        return g

    def _prepare_instances(C, instance_directory: pathlib.Path, cache_directory, **kwargs):
        cache_directory.mkdir(parents=True, exist_ok=True)
        weighted = kwargs.get("weighted", False)
        dest_graphs_file = cache_directory / f"graphs_{'weighted' if weighted else 'unweighted'}.dgl"
        name_mapping_file = cache_directory / f"graph_names.json"
        last_updated = 0
        if os.path.exists(str(dest_graphs_file)):
            logger.info(f"Found existing graphs file {dest_graphs_file}")
            last_updated = os.path.getmtime(dest_graphs_file)
            gs, _ = dgl.load_graphs(str(dest_graphs_file))

        for graph_path in instance_directory.rglob("*.gpickle"):
            graph_file = graph_path.resolve()
            source_mtime = os.path.getmtime(graph_file)
            if source_mtime > last_updated or not os.path.exists(name_mapping_file):
                logger.info(f"Updated graph file: {graph_file} (or name mapping was not existing yet).")
                logger.info(f"Re-converting all graphs... this can take a while.")
                gs = []
                graph_names = []
                for graph_path in tqdm(instance_directory.rglob("*.gpickle")):
                    graph_file = graph_path.resolve()
                    nx_graph = nx.read_gpickle(graph_file)
                    g = C.__prepare_graph(nx_graph, weighted=weighted)
                    gs.append(g)
                    graph_names.append(os.path.splitext(os.path.basename(graph_file))[0])
                dgl.save_graphs(str(dest_graphs_file), gs)
                with open(name_mapping_file, "w", encoding='utf-8') as f:
                    json.dump(graph_names, f, ensure_ascii=False, sort_keys = True, indent=4)
                break

    def train(self, train_data_path: pathlib.Path, results_path: pathlib.Path, parameters):
        cache_directory = train_data_path / "preprocessed" / str(self)

        if "weighted" in parameters.keys():
            weighted = True
        else:
            weighted = False

        self._prepare_instances(train_data_path, cache_directory, weighted=weighted)

        logger.info("Invoking training of " + str(self))
        conda_env_name = str(self)
        conda_env_file = self.directory() / "environment.yml"
        script_file = self.directory() / "main.py"
        cuda_devices = ' '.join(map(str, parameters["cuda_devices"]))

        arguments = [
            "train", # operation
            cache_directory, # input
            results_path, # output
            "--loglevel", parameters["loglevel"]
        ]
        
        if "self_loops" in parameters.keys():
            arguments += ["--self_loops"]

        if "weighted" in parameters.keys():
            arguments += ["--weighted"]

        if self.model_state_path:
            arguments += ["--pretrained_weights", self.model_state_path]

        if parameters["cuda_devices"]:
            arguments += ["--cuda_devices", cuda_devices]

        if "model_prob_maps" in parameters.keys():
            arguments += ["--model_prob_maps", parameters["model_prob_maps"]]

        if "epochs" in parameters.keys():
            arguments += ["--epochs", parameters["epochs"]]

        if "lr" in parameters.keys():
            arguments += ["--lr", parameters["lr"]]

        logger.debug(f"Calling {script_file} with arguments {arguments}.")

        launch_python_script_in_conda_env(conda_env_name, conda_env_file, script_file, arguments)

    def solve(self, solve_data_path: pathlib.Path, results_path: pathlib.Path, parameters):
        if not self.model_state_path:
            raise ValueError("Cannot use solver without trained weights.")

        if "weighted" in parameters.keys():
            weighted = True
        else:
            weighted = False

        cache_directory = solve_data_path / "preprocessed" / str(self)
        self._prepare_instances(solve_data_path, cache_directory, weighted=weighted)

        logger.info("Solving all given instances using " + str(self))
        conda_env_name = str(self)
        conda_env_file = self.directory() / "environment.yml"
        script_file = self.directory() / "main.py"

        cuda_devices = list(map(str, parameters["cuda_devices"]))

        arguments = [
            "solve", # operation
            cache_directory, # input
            results_path, # output
            "--num_threads", parameters["num_threads"],
            "--time_limit", parameters["time_limit"],
            "--pretrained_weights", self.model_state_path,
            "--loglevel", parameters["loglevel"]
        ]

        if "self_loops" in parameters.keys():
            arguments += ["--self_loops"]

        if "max_prob_maps" in parameters.keys():
            arguments += ["--max_prob_maps", parameters["max_prob_maps"]]

        if "model_prob_maps" in parameters.keys():
            arguments += ["--model_prob_maps", parameters["model_prob_maps"]]

        if parameters["cuda_devices"]:
            arguments += ["--cuda_devices"]
            arguments += cuda_devices

        if "reduction" in parameters.keys():
            arguments += ["--reduction"]

        if "local_search" in parameters.keys():
            arguments += ["--local_search"]

        if "queue_pruning" in parameters.keys():
            arguments += ["--queue_pruning"]

        if weighted:
            arguments += ["--weighted"]

        if "noise_as_prob_maps" in parameters.keys():
            arguments += ["--noise_as_prob_maps"]

        if "weighted_queue_pop" in parameters.keys():
            arguments += ["--weighted_queue_pop"]

        logger.debug(f"Calling {script_file} with arguments {arguments}.")

        launch_python_script_in_conda_env(conda_env_name, conda_env_file, script_file, arguments)
