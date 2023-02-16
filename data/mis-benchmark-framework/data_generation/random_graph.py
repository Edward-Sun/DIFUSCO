from data_generation.generator import DataGenerator
import networkx as nx
import random
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
import subprocess
import os
from logzero import logger
import tqdm
from utils import run_command_with_live_output

class GraphSampler(ABC):
    @abstractmethod
    def generate_graph(self):
        pass


class ErdosRenyi(GraphSampler):
    def __init__(self, min_n, max_n, p):
        self.min_n = min_n
        self.max_n = max_n
        self.p = p

    def __str__(self):
        return f"ER_{self.min_n}_{self.max_n}_{self.p}"

    def generate_graph(self):
        n = random.randint(self.min_n, self.max_n)
        return nx.erdos_renyi_graph(n, self.p)


class BarabasiAlbert(GraphSampler):
    def __init__(self, min_n, max_n, m):
        self.min_n = min_n
        self.max_n = max_n
        self.m = m

    def __str__(self):
        return f"BA_{self.min_n}_{self.max_n}_{self.m}"

    def generate_graph(self):
        n = random.randint(self.min_n, self.max_n)
        return nx.barabasi_albert_graph(n, min(self.m, n))


class HolmeKim(GraphSampler):
    def __init__(self, min_n, max_n, m, p):
        self.min_n = min_n
        self.max_n = max_n
        self.m = m
        self.p = p

    def __str__(self):
        return f"HK_{self.min_n}_{self.max_n}_{self.m}_{self.p}"

    def generate_graph(self):
        n = random.randint(self.min_n, self.max_n)
        return nx.powerlaw_cluster_graph(n, min(self.m, n), self.p)


class WattsStrogatz(GraphSampler):
    def __init__(self, min_n, max_n, k, p):
        self.min_n = min_n
        self.max_n = max_n
        self.k = k
        self.p = p

    def __str__(self):
        return f"WS_{self.min_n}_{self.max_n}_{self.k}_{self.p}"

    def generate_graph(self):
        n = random.randint(self.min_n, self.max_n)
        return nx.watts_strogatz_graph(n, self.k, self.p)


class HyperbolicRandomGraph(GraphSampler):
    def __init__(self, min_n, max_n, alpha, t, degree, threads):
        self.min_n = min_n
        self.max_n = max_n
        self.alpha = alpha
        self.t = t
        self.degree = degree
        self.threads = threads

        girgs_path = Path(__file__).parent / "girgs"

        if not girgs_path.exists():
            girgs_repo = "https://github.com/chistopher/girgs"
            target_commit = "c38e4118f02cffae51b1eaf7a1c1f9314a6a89c8"
            subprocess.run(["git", "clone", girgs_repo], cwd=Path(__file__).parent)
            subprocess.run(["git","checkout", target_commit], cwd=girgs_path)
            os.mkdir(girgs_path / "build")
            subprocess.run(["cmake", ".."], cwd=girgs_path / "build")
            subprocess.run(["make", "genhrg"], cwd=girgs_path / "build")

        self.binary_path = girgs_path / "build" / "genhrg"
        self.tmp_path = girgs_path

    def __str__(self):
        return f"HRG_{self.min_n}_{self.max_n}_{self.alpha}_{self.t}_{self.degree}"

    def generate_graph(self):
        n = random.randint(self.min_n, self.max_n)
        command = [self.binary_path, "-n", str(n), "-alpha", str(self.alpha), "-t", str(self.t), "-deg", str(self.degree), "-threads", str(self.threads), "-edge", "1", "-file", str(self.tmp_path / "tmp")]
        run_command_with_live_output(command)

        with open(self.tmp_path / "tmp.txt", 'r') as file:
            content = file.read().split('\n')

        edge_list = list(map(lambda x: tuple(map(int, x.split())), content[2:-1]))
        logger.debug(f"edge_list = {edge_list}")
        G = nx.empty_graph(n)
        G.add_edges_from(edge_list)
        logger.debug(f"Generated HRG with {G.number_of_nodes()} nodes (n = {n}).")
        os.remove(self.tmp_path / "tmp.txt")

        return G

class RandomGraphGenerator(DataGenerator):
    def __init__(self, output_path, graph_sampler: GraphSampler, num_graphs = 1):
        self.num_graphs = num_graphs
        self.output_path = output_path
        self.graph_sampler = graph_sampler

    def generate(self, gen_labels = False, weighted = False):
        for i in tqdm.tqdm(range(self.num_graphs)):
            stub = f"{self.graph_sampler}_{i}"
            G = self.graph_sampler.generate_graph()

            if weighted:
                weight_mapping = { vertex: int(weight) for vertex, weight in zip(G.nodes, self.random_weight(G.number_of_nodes(), sigma=30, mu=100)) }
                nx.set_node_attributes(G, values = weight_mapping, name='weight')

            if gen_labels:
                mis, status = self._call_gurobi_solver(G, weighted=weighted)
                label_mapping = { vertex: int(vertex in mis) for vertex in G.nodes }
                nx.set_node_attributes(G, values = label_mapping, name='label' if status == "Optimal" else 'nonoptimal_label')

                if status != "Optimal":
                    logger.warn(f"Graph {i} has non-optimal labels (mis size = {len(mis)})!")

            output_file = self.output_path / (f"{stub}{'.non-optimal' if gen_labels and status != 'Optimal' else ''}.gpickle")
            with open(output_file, "wb") as f:
                pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)
