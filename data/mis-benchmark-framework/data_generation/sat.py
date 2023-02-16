from data_generation.generator import DataGenerator
from pathlib import Path

from pysat.formula import CNF
import networkx as nx
import numpy as np
import pickle

class SATGraphDataGenerator(DataGenerator):

    def __init__(self, input_path, output_path):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)

    def _build_graph(self, cnf_file, output_file, gen_labels, weighted):
        cnf = CNF(cnf_file)
        nv = cnf.nv
        clauses = list(filter(lambda x: x, cnf.clauses))
        ind = { k:[] for k in np.concatenate([np.arange(1, nv+1), -np.arange(1, nv+1)]) }
        edges = []
        for i, clause in enumerate(clauses):
            a = clause[0]
            b = clause[1]
            c = clause[2]
            aa = 3 * i + 0
            bb = 3 * i + 1
            cc = 3 * i + 2
            ind[a].append(aa)
            ind[b].append(bb)
            ind[c].append(cc)
            edges.append((aa, bb))
            edges.append((aa, cc))
            edges.append((bb, cc))

        for i in np.arange(1, nv+1):
            for u in ind[i]:
                for v in ind[-i]:
                    edges.append((u, v))

        G = nx.from_edgelist(edges)

        if gen_labels:
            mis = self._call_gurobi_solver(G, use_multiprocessing=True)[0]
            label_mapping = { vertex: int(vertex in mis) for vertex in G.nodes }
            # print("label_mapping", label_mapping)
            nx.set_node_attributes(G, values=label_mapping, name='label')

        if weighted:
            weight_mapping = {vertex: weight for vertex, weight in zip(G.nodes, self.random_weight(G.number_of_nodes()))}
            nx.set_node_attributes(G, values=weight_mapping, name='weight')

        # write graph object to output file
        with open(output_file, "wb") as f:
            pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)

    def generate(self, gen_labels=False,  weighted=False):
        # for f in self.input_path.rglob("*.cnf"):
        #     self._build_graph(f, self.output_path / (f.stem + ".gpickle"), gen_labels, weighted)
        import functools
        imap_unordered_bar(functools.partial(self.func, gen_labels=gen_labels, weighted=weighted),
                           self.input_path.rglob("*.cnf"), n_processes=2)

    def func(self, f, gen_labels, weighted):
        self._build_graph(f, self.output_path / (f.stem + ".gpickle"), gen_labels, weighted)


from multiprocessing import Pool
from tqdm import *


def imap_unordered_bar(func, args, n_processes=2):
    p = Pool(n_processes)
    args = list(args)
    # print(args)
    with tqdm(total=len(args)) as pbar:
        for i, res in tqdm(enumerate(p.imap_unordered(func, args))):
            pbar.update()
    pbar.close()
    p.close()
    p.join()
