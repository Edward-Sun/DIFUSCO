import subprocess
import os.path
import pathlib

import json
import numpy as np
import networkx as nx
import dgl
import time
import re

from logzero import logger
from utils import run_command_with_live_output
from solvers.abstractsolver import MWISSolver

class KaMIS(MWISSolver):
    def __init__(self) -> None:
        kamis_path = self.directory() / "KaMIS"

        if not kamis_path.exists():
            kamis_repo = "https://github.com/KarlsruheMIS/KaMIS"
            target_commit = "791334ef6aebb2ccf005316258c043cca58923a0"
            subprocess.run(["git", "clone", kamis_repo], cwd=self.directory())
            subprocess.run(["git","checkout",target_commit], cwd=kamis_path)
            subprocess.run(["bash", "compile_withcmake.sh"], cwd=kamis_path)

    def load_weights(self, model_state_path):
        raise NotImplementedError("KaMIS requires no weights.")

    def __str__(self) -> str:
        return "kamis"

    def directory(self):
        return pathlib.Path(__file__).parent / "kamis"

    @staticmethod
    def __prepare_graph(g: nx.Graph(), weighted = False):
        g.remove_edges_from(nx.selfloop_edges(g))
        n = g.number_of_nodes()
        m = g.number_of_edges()
        wt = 0 if not weighted else 10

        res = f"{n} {m} {wt}\n"

        for n, nbrsdict in g.adjacency():
            line = []
            
            #if weighted: line.append(g.node[n]["weight"]
            if weighted:
                line.append(g.nodes(data="weight", default=1)[n])

            for nbr, _ in sorted(nbrsdict.items()):
                line.append(nbr + 1)
            res += " ".join(map(str, line)) + "\n"
        return res


    def _prepare_instance(source_instance_file: pathlib.Path, cache_directory: pathlib.Path, weighted=False):
        cache_directory.mkdir(parents=True, exist_ok=True)

        dest_path = cache_directory / (source_instance_file.stem + f"_{'weighted' if weighted else 'unweighted'}.graph")
        
        if os.path.exists(dest_path):
            source_mtime = os.path.getmtime(source_instance_file)
            last_updated = os.path.getmtime(dest_path)

            if source_mtime <= last_updated:
                return # we already have an up2date version of that file

        logger.info(f"Updated graph file: {source_instance_file}.")
        
        g = nx.read_gpickle(source_instance_file)
        graph = KaMIS.__prepare_graph(g, weighted=weighted)
        
        with open(dest_path, "w") as res_file:
            res_file.write(graph)

    def train(self, train_data_path: pathlib.Path, results_path: pathlib.Path, parameters):
        raise NotImplementedError("KaMIS cannot be trained!")

    def solve(self, solve_data_path: pathlib.Path, results_path: pathlib.Path, parameters):
        logger.info("Solving all given instances using " + str(self))

        weighted = "weighted" in parameters.keys()
        cache_directory = solve_data_path / "preprocessed" / str(self)
        self._prepare_instances(solve_data_path, cache_directory, weighted=weighted)

        # for graph_path in solve_data_path.rglob("*.gpickle"):

        import functools
        # def solve_graph(graph_path, weighted, directory, cache_directory, results_path, parameters):
        argumented_solve_graph = functools.partial(
            solve_graph,
            weighted=weighted,
            directory=self.directory(),
            cache_directory=cache_directory,
            results_path=results_path,
            parameters=parameters)

        res_list = imap_unordered_bar(argumented_solve_graph, solve_data_path.rglob("*.gpickle"), n_processes=64)

        results = {}
        for res in res_list:
            results.update(res)

        with open(results_path / "results.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, sort_keys=True, indent=4)


from multiprocessing import Pool
from tqdm import tqdm


def imap_unordered_bar(func, args, n_processes=2):
    p = Pool(n_processes)
    args = list(args)
    res_list = []
    with tqdm(total=len(args)) as pbar:
        for i, res in tqdm(enumerate(p.imap_unordered(func, args))):
            pbar.update()
            res_list.append(res)
    pbar.close()
    p.close()
    p.join()
    return res_list


def solve_graph(graph_path, weighted, directory, cache_directory, results_path, parameters):
    results = {}
    if weighted:
        executable = directory / "KaMIS" / "deploy" / "weighted_branch_reduce"
    else:
        executable = directory / "KaMIS" / "deploy" / "redumis"

    _preprocessed_graph = cache_directory / (graph_path.stem + f"_{'weighted' if weighted else 'unweighted'}.graph")

    results_filename = results_path / (graph_path.stem + f"_{'weighted' if weighted else 'unweighted'}.result")

    pass_kamis = False
    if os.path.exists(results_filename):
        try:
            with open(results_filename, "r") as f:
                vertices = list(map(int, f.read().replace('\n', '')))
            if len(vertices) > 0:
                pass_kamis = True
        except (FileNotFoundError, ValueError):
            pass_kamis = False

    if not pass_kamis:
        arguments = [
            _preprocessed_graph,  # input
            "--output", results_filename,  # output
            "--time_limit", str(parameters["time_limit"]),
        ]

        logger.debug(f"Calling {executable} with arguments {arguments}.")
        start_time = time.monotonic()
        # out = subprocess.run(executable=executable, args=arguments, capture_output=True, text=True)
        # stdout = out.stdout
        _, lines = run_command_with_live_output([executable] + arguments, capture_output=True)
        solve_time = time.monotonic() - start_time

        results[graph_path.stem] = {"total_time": solve_time}
        try:
            with open(results_filename, "r") as f:
                vertices = list(map(int, f.read().replace('\n', '')))
            is_vertices = np.flatnonzero(np.array(vertices))
        except (FileNotFoundError, ValueError):
            print("\n".join(lines))
            return results

        if weighted:
            discovery = re.compile("^(\d+(\.\d*)?) \[(\d+\.\d*)\]$")
            max_mwis_weight = 0
            mis_time = 0.0
            for line in lines:
                match = discovery.match(line)
                if match:
                    mwis_weight = float(match[1])
                    if mwis_weight > max_mwis_weight:
                        max_mwis_weight = mwis_weight
                        mis_time = float(match[3])

            if max_mwis_weight == 0:
                # try another method
                for line in lines:
                    if line.startswith("time"):
                        mis_time = line.split(" ")[1]

                    if line.startswith("MIS_weight"):
                        max_mwis_weight = line.split(" ")[1]

            if max_mwis_weight == 0:
                results[graph_path.stem]["mwis_found"] = False
            else:
                results[graph_path.stem]["mwis_found"] = True
                results[graph_path.stem]["mwis"] = is_vertices.tolist()
                results[graph_path.stem]["time_to_find_mwis"] = mis_time
                results[graph_path.stem]["mwis_vertices"] = is_vertices.shape[0]
                results[graph_path.stem]["mwis_weight"] = max_mwis_weight

        else:
            stdout = "\n".join(lines)
            discovery = re.compile("Best solution:\s+(\d+)\nTime:\s+(\d+\.\d*)\n", re.MULTILINE)

            time_found_in_stdout = False

            solution_time = 0.0
            for size, timestamp in discovery.findall(stdout):
                if int(size) == is_vertices.shape[0]:
                    solution_time = float(timestamp)
                    time_found_in_stdout = True
                    break

            if not time_found_in_stdout:
                # try another regex
                discovery = re.compile("Best\n={42}\nSize:\s+\d+\nTime found:\s+(\d+\.\d*)", re.MULTILINE)
                m = discovery.search(stdout)
                if m:
                    solution_time = float(m.group(1))
                    time_found_in_stdout = True

            if not time_found_in_stdout:
                results[graph_path.stem]["found_mis"] = False
            else:
                results[graph_path.stem]["found_mis"] = True
                results[graph_path.stem]["mis"] = is_vertices.tolist()
                results[graph_path.stem]["vertices"] = is_vertices.shape[0]
                results[graph_path.stem]["solution_time"] = solution_time

    return results
