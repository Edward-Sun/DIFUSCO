from re import split
from data_generation.generator import DataGenerator
from pathlib import Path

from pysat.formula import CNF
import networkx as nx
import numpy as np
import subprocess
import tempfile
import shutil
from pathlib import Path
import networkx as nx
import pandas as pd
import scipy.io
import sys
from logzero import logger

def url_to_filename(url):
    return url.split("/")[-1]

def is_module_available(module_name):
    if sys.version_info < (3, 0):
        # python 2
        import importlib
        torch_loader = importlib.find_loader(module_name)
    elif sys.version_info <= (3, 3):
        # python 3.0 to 3.3
        import pkgutil
        torch_loader = pkgutil.find_loader(module_name)
    elif sys.version_info >= (3, 4):
        # python 3.4 and above
        import importlib
        torch_loader = importlib.util.find_spec(module_name)
    return torch_loader is not None

class GraphDataset(object):
    def __init__(self, url, unpack=True):
        self.url = url
        self.unpack = unpack

    def __enter__(self, ):
        self.dir = tempfile.mkdtemp()
        p = subprocess.Popen(["wget", self.url], cwd=self.dir)
        p.wait()
        
        if self.unpack:
            p = subprocess.Popen(["unp", url_to_filename(self.url)], cwd=self.dir)
            p.wait()

        return Path(self.dir)

    def __exit__(self, type, value, traceback):
        shutil.rmtree(self.dir)

def clean_nx_graph(G):
    _G = nx.convert_node_labels_to_integers(nx.Graph(nx.to_undirected(G)))
    _G.remove_edges_from(nx.selfloop_edges(_G))
    return _G

class RealWorldGraphGenerator(DataGenerator):

    def __init__(self, output_path, limit):
        self.output_path = Path(output_path)
        self.limit = limit

        if self.limit:
            logger.info(f"Set limit for RWG to {self.limit}")

    def get_dataset_directory(self, dataset_name):
        return self.output_path / dataset_name

    def create_if_needed(self, dataset_name):
        dataset_directory = self.get_dataset_directory(dataset_name)
        if not dataset_directory.exists():
            dataset_directory.mkdir()
            return True
        return False

    def handle_amazon(self, name, file_list, gen_labels):
        if self.create_if_needed(name):
            amzn_base = "https://mwis-vr-instances.s3.amazonaws.com/"

            for graph_idx,file in enumerate(file_list):
                file_url = amzn_base + file
                with GraphDataset(file_url) as dsdir:
                    df = pd.read_csv(dsdir / file.replace(".tar.gz", "") / "conflict_graph.txt", sep=" ", header=None, skiprows=1, names=["source", "target"])

                    G = nx.from_pandas_edgelist(df)
                    weight_mapping = { }
                    with open(dsdir / file.replace(".tar.gz", "")  / "node_weights.txt", "r") as a_file:
                        for line in a_file:
                            stripped_line = line.strip()
                            splitted = stripped_line.split()
                            weight_mapping[int(splitted[0])] = int(splitted[1])

                    nx.set_node_attributes(G, values = weight_mapping, name='weight')

                    G = clean_nx_graph(G)
                    G = self.maybe_label_graph(gen_labels, G, f"{name}-{graph_idx}")

                    nx.write_gpickle(G, str(self.get_dataset_directory(name) / f"{name}-{graph_idx}.gpickle"))
                    logger.info(f"Saved {name}-{graph_idx}. nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")

    def handle_reddit(self, name, fullname, url, gen_labels):
        if self.create_if_needed(name):
            with GraphDataset(url) as dsdir:
                node_to_graph_dict = {}
                with open(dsdir / fullname / f"{fullname}_graph_indicator.txt") as file:
                    for idx,line in enumerate(file):
                        graph = int(line.strip()) - 1 # 0-indexed graph mapping
                        node_to_graph_dict[idx+1] = graph # 1-indexed nodes

                graph_edgelists = {}
                with open(dsdir / fullname / f"{fullname}_A.txt") as file:
                    for line in file:
                        _line = list(map(lambda s: int(s), line.strip().split(", "))) # space is there on purpose!
                        u = _line[0]
                        v = _line[1]
                        if node_to_graph_dict[u] != node_to_graph_dict[v]:
                            logger.info(f"Error u={u}, v={v}, graph(u) = { node_to_graph_dict[u] }, graph(v) = { node_to_graph_dict[v] }")
                        graph_idx = node_to_graph_dict[u]

                        if graph_idx not in graph_edgelists.keys():
                            graph_edgelists[graph_idx] = []

                        graph_edgelists[graph_idx].append(f"{u} {v}")

                for i,graph_idx in enumerate(graph_edgelists.keys()):
                    if self.limit is not None and i >= self.limit:
                        logger.info(f"Reached limit of {self.limit} graphs, done with {name}.")
                        break

                    G = nx.parse_edgelist(graph_edgelists[graph_idx], nodetype=int)
                    G = clean_nx_graph(G)
                    G = self.maybe_label_graph(gen_labels, G, f"{name}-{graph_idx}")

                    nx.write_gpickle(G, str(self.get_dataset_directory(name) / f"{name}-{graph_idx}.gpickle"))
                    logger.info(f"Saved {name}-{graph_idx}. nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")

    def maybe_label_graph(self, gen_labels, G, name):
        if gen_labels:
            mis, status = self._call_gurobi_solver(G, timeout=120)
            label_mapping = { vertex: int(vertex in mis) for vertex in G.nodes }
            nx.set_node_attributes(G, values = label_mapping, name='label' if status == "Optimal" else 'nonoptimal_label')

            if status != "Optimal":
                logger.warn(f"Graph {name} has non-optimal labels (mis size = {len(mis)})!")

        return G

    def generate(self, gen_labels = False,  weighted = False):
    
        self.handle_reddit("reddit-b", "REDDIT-BINARY", "https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/REDDIT-BINARY.zip", gen_labels)
        self.handle_reddit("reddit-5k", "REDDIT-MULTI-5K", "https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/REDDIT-MULTI-5K.zip", gen_labels)
        self.handle_reddit("reddit-12k", "REDDIT-MULTI-12K", "https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/REDDIT-MULTI-12K.zip", gen_labels)

        self.handle_amazon("amazon-mt", ["MT-D-01.tar.gz", "MT-D-200.tar.gz", "MT-D-FN.tar.gz", "MT-W-01.tar.gz", "MT-W-200.tar.gz", "MT-W-FN.tar.gz"], gen_labels)
        self.handle_amazon("amazon-mr", ["MR-D-01.tar.gz", "MR-D-03.tar.gz", "MR-D-05.tar.gz", "MR-D-FN.tar.gz", "MR-W-FN.tar.gz"], gen_labels)
        self.handle_amazon("amazon-mw", ["MW-D-01.tar.gz", "MW-D-20.tar.gz", "MW-D-40.tar.gz", "MW-D-FN.tar.gz", "MW-W-01.tar.gz", "MW-W-05.tar.gz", "MW-W-10.tar.gz", "MW-W-FN.tar.gz"], gen_labels)
        self.handle_amazon("amazon-cw", ["CW-T-C-1.tar.gz", "CW-T-C-2.tar.gz", "CW-T-D-4.tar.gz", "CW-T-D-6.tar.gz"], gen_labels)
        self.handle_amazon("amazon-cr", ["CR-T-C-1.tar.gz", "CR-T-C-2.tar.gz", "CR-T-D-4.tar.gz", "CR-T-D-6.tar.gz", "CR-T-D-7.tar.gz"], gen_labels)

        if self.create_if_needed("as-caida"):
            with GraphDataset("https://snap.stanford.edu/data/as-caida.tar.gz") as dsdir:
                for idx,graph_path in enumerate(dsdir.rglob("*.txt")):
                    if self.limit is not None and idx >= self.limit:
                        logger.info(f"Reached limit of {self.limit} graphs, done with as-caida.")
                        break

                    df = pd.read_csv(graph_path, skiprows = 8, sep="\t", header=None, names=["source", "target", "label"])
                    G = clean_nx_graph(nx.from_pandas_edgelist(df))
                    G = self.maybe_label_graph(gen_labels, G, f"as-caida-{idx}")
                    nx.write_gpickle(G, str(self.get_dataset_directory("as-caida") / f"as-caida-{idx}.gpickle"))
                    logger.info(f"Saved as-caida-{idx}. nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")

        if self.create_if_needed("citeseer"):
            with GraphDataset("https://nrvis.com/download/data/labeled/citeseer.zip") as dsdir:
                df = pd.read_csv(dsdir / "citeseer.edges", header=None, names=["source", "target", "label"])
                G = clean_nx_graph(nx.from_pandas_edgelist(df))
                G = self.maybe_label_graph(gen_labels, G, "citeseer")
                nx.write_gpickle(G, str(self.get_dataset_directory("citeseer") / "citeseer.gpickle"))
                logger.info(f"Saved citeseer. nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")

        if self.create_if_needed("cora"):
            with GraphDataset("https://nrvis.com/download/data/labeled/cora.zip") as dsdir:
                G = clean_nx_graph(nx.from_pandas_edgelist(pd.read_csv(dsdir / "cora.edges", header=None, names=["target", "source", "label"])))
                G = self.maybe_label_graph(gen_labels, G, "cora")
                nx.write_gpickle(G, str(self.get_dataset_directory("cora") / "cora.gpickle"))
                logger.info(f"Saved cora. nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")

        if self.create_if_needed("PubMed"):
            with GraphDataset("https://nrvis.com/download/data/labeled/PubMed.zip") as dsdir:
                df = pd.read_csv(dsdir / "PubMed.edges", header=None, names=["source", "target"])
                G = clean_nx_graph(nx.from_pandas_edgelist(df))
                G = self.maybe_label_graph(gen_labels, G, "PubMed")
                nx.write_gpickle(G, str(self.get_dataset_directory("PubMed") / "PubMed.gpickle"))
                logger.info(f"Saved PubMed. nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")

        if self.create_if_needed("dblp"):
            with GraphDataset("https://snap.stanford.edu/data/bigdata/communities/com-dblp.ungraph.txt.gz") as dsdir:
                df = pd.read_csv(dsdir / "com-dblp.ungraph.txt", header=None, names=["source", "target"], sep="\t", skiprows=4)
                G = clean_nx_graph(nx.from_pandas_edgelist(df))
                G = self.maybe_label_graph(gen_labels, G, "dblp")
                nx.write_gpickle(G, str(self.get_dataset_directory("dblp") / "dblp.gpickle"))
                logger.info(f"Saved DBLP. nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")

        if self.create_if_needed("wiki-Vote"):
            with GraphDataset("https://snap.stanford.edu/data/wiki-Vote.txt.gz") as dsdir:
                df = pd.read_csv(dsdir / "wiki-Vote.txt", header=None, names=["source", "target"], sep="\t", skiprows=4)
                G = clean_nx_graph(nx.from_pandas_edgelist(df))
                G = self.maybe_label_graph(gen_labels, G, "wiki-Vote")
                nx.write_gpickle(G, str(self.get_dataset_directory("wiki-Vote") / "wiki-Vote.gpickle"))
                logger.info(f"Saved wiki-Vote. nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")

        if self.create_if_needed("wiki-RfA"):
            with GraphDataset("https://suitesparse-collection-website.herokuapp.com/MM/SNAP/wiki-RfA.tar.gz") as dsdir:
                mat = scipy.io.mmread(dsdir / "wiki-RfA/wiki-RfA.mtx")
                G = clean_nx_graph(nx.from_scipy_sparse_matrix(mat))
                G = self.maybe_label_graph(gen_labels, G, "wiki-RfA")
                nx.write_gpickle(G, str(self.get_dataset_directory("wiki-RfA") / "wiki-RfA.gpickle"))
                logger.info(f"Saved wiki-RfA. nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")

        if self.create_if_needed("bitcoin-otc"):
            with GraphDataset("https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz") as dsdir:
                df = pd.read_csv(dsdir / "soc-sign-bitcoinotc.csv", header=None, names=["source", "target", "rating", "time"])
                G = clean_nx_graph(nx.from_pandas_edgelist(df))
                G = self.maybe_label_graph(gen_labels, G, "bitcoin-otc")
                nx.write_gpickle(G, str(self.get_dataset_directory("bitcoin-otc") / "bitcoin-otc.gpickle"))
                logger.info(f"Saved bitcoin-otc. nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")

        if self.create_if_needed("bitcoin-alpha"):
            with GraphDataset("https://snap.stanford.edu/data/soc-sign-bitcoinalpha.csv.gz") as dsdir:
                df = pd.read_csv(dsdir / "soc-sign-bitcoinalpha.csv", header=None, names=["source", "target", "rating", "time"])
                G = clean_nx_graph(nx.from_pandas_edgelist(df))
                G = self.maybe_label_graph(gen_labels, G, "bitcoin-alpha")
                nx.write_gpickle(G, str(self.get_dataset_directory("bitcoin-alpha") / "bitcoin-alpha.gpickle"))
                logger.info(f"Saved bitcoin-alpha. nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")

        if self.create_if_needed("roadnet-pennsylvania"):
            with GraphDataset("https://snap.stanford.edu/data/roadNet-PA.txt.gz") as dsdir:
                df = pd.read_csv(dsdir / "roadNet-PA.txt", sep="\t", header=None, skiprows=4, names=["source", "target"])
                G = clean_nx_graph(nx.from_pandas_edgelist(df))
                G = self.maybe_label_graph(gen_labels, G, "roadnet-pennsylvania")
                nx.write_gpickle(G, str(self.get_dataset_directory("roadnet-pennsylvania") / "roadnet-pennsylvania.gpickle"))
                logger.info(f"Saved roadnet-pennsylvania. nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")

        if self.create_if_needed("web-google"):
            with GraphDataset("https://snap.stanford.edu/data/web-Google.txt.gz") as dsdir:
                df = pd.read_csv(dsdir / "web-Google.txt", sep="\t", header=None, skiprows=4, names=["source", "target"])
                G = clean_nx_graph(nx.from_pandas_edgelist(df))
                G = self.maybe_label_graph(gen_labels, G, "web-google")
                nx.write_gpickle(G, str(self.get_dataset_directory("web-google") / "web-google.gpickle"))
                logger.info(f"Saved web-google. nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")

        if self.create_if_needed("ego-gplus"):
            with GraphDataset("https://snap.stanford.edu/data/gplus_combined.txt.gz") as dsdir:
                df = pd.read_csv(dsdir / "gplus_combined.txt", sep=" ", header=None, names=["source", "target"])
                G = clean_nx_graph(nx.from_pandas_edgelist(df))
                G = self.maybe_label_graph(gen_labels, G, "ego-gplus")
                nx.write_gpickle(G, str(self.get_dataset_directory("ego-gplus") / "ego-gplus.gpickle"))
                logger.info(f"Saved ego-gplus. nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")

        if self.create_if_needed("ego-facebook"):
            with GraphDataset("https://snap.stanford.edu/data/facebook_combined.txt.gz") as dsdir:
                df = pd.read_csv(dsdir / "facebook_combined.txt", sep=" ", header=None, names=["source", "target"])
                G = clean_nx_graph(nx.from_pandas_edgelist(df))
                G = self.maybe_label_graph(gen_labels, G, "ego-facebook")
                nx.write_gpickle(G, str(self.get_dataset_directory("ego-facebook") / "ego-facebook.gpickle"))
                logger.info(f"Saved ego-facebook. nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")

        if self.create_if_needed("ppi"):
            if not is_module_available("dgl"):
                logger.info("Cannot load ppi dataset without dgl available! Skipping.")
            else:
                import dgl
                graphs = []
                output_folder = self.get_dataset_directory("ppi")
                for mode in ["train", "valid", "test"]:
                    data = dgl.data.ppi.PPIDataset(mode)
                    if not data.has_cache():
                        data.download()
                        data.process()
                        data.save()
                    data.load()
                    graphs.extend(data.graphs)

                nx_graphs = [clean_nx_graph(dgl.to_networkx(G)) for G in graphs]

                for idx,G in enumerate(nx_graphs):
                    if self.limit is not None and idx >= self.limit:
                        logger.info(f"Reached limit of {self.limit} graphs, done with ppi.")
                        break

                    G = self.maybe_label_graph(gen_labels, G, f"ppi-{idx}")
                    nx.write_gpickle(G, f"{output_folder}/ppi-{idx}.gpickle")
                    logger.info(f"Saved ppi-{idx}. nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")

        if self.create_if_needed("roadnet-berlin"):
            if not is_module_available("osmnx") or not is_module_available("dgl"):
                logger.info("Cannot load roadnet-berlin dataset without osmnx and dgl available! Skipping.")
            else:
                import osmnx as ox
                import dgl
                ox.config(use_cache=True, log_console=True)
                G = ox.graph_from_bbox(52.6989, 52.1912, 14.1930, 12.7208, network_type="drive")
                G = dgl.from_networkx(G) # need to do this, otherwise nx graph is dirty from osmnx stuff
                G = clean_nx_graph(dgl.to_networkx(G))
                G = self.maybe_label_graph(gen_labels, G, "roadnet-berlin")
                nx.write_gpickle(G, str(self.get_dataset_directory("roadnet-berlin") / "roadnet-berlin.gpickle"))
                logger.info(f"Saved roadnet-berlin. nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")

        if self.create_if_needed("kexu-vc-benchmark"):
            idx = 0
            for variant in ["30-15", "35-17", "40-19", "45-21", "50-23", "53-24", "56-25", "59-26"]:
                for instance in range(1,6):
                    with GraphDataset(f"https://raw.githubusercontent.com/unsat/npbench/master/instances/vertex_cover/benchmarks/frb{variant}-mis/frb{variant}-{instance}.mis", unpack=False) as dsdir:
                        df = pd.read_csv(dsdir / f"frb{variant}-{instance}.mis", sep=" ", header=None, skiprows=1, names=["p", "source", "target"])
                        G = clean_nx_graph(nx.from_pandas_edgelist(df))
                        G = self.maybe_label_graph(gen_labels, G, f"kexu-vc-benchmark-{idx}")
                        nx.write_gpickle(G, str(self.get_dataset_directory("kexu-vc-benchmark") / f"kexu-vc-benchmark-{idx}.gpickle"))
                        logger.info(f"Saved kexu-vc-benchmark-{idx}. nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")
                        idx += 1

        if self.create_if_needed("dimacs"):
            with GraphDataset(f"http://lcs.ios.ac.cn/~caisw/Resource/DIMACS complementary graphs.tar.gz", unpack=True) as dsdir:
                for idx, graph_path in enumerate(dsdir.rglob("*.mis")):
                    df = pd.read_csv(graph_path, sep=" ", header=None, skiprows=1, names=["p", "source", "target"])
                    G = clean_nx_graph(nx.from_pandas_edgelist(df))
                    G = self.maybe_label_graph(gen_labels, G, f"dimacs-{idx}")
                    nx.write_gpickle(G, str(self.get_dataset_directory("dimacs") / f"dimacs-{idx}.gpickle"))
                    logger.info(f"Saved dimacs-{idx}. nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")
