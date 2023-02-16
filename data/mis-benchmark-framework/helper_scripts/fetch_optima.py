#!/usr/bin/env python3

"""
Evaluating the results of different solvers requires us to (sometimes) compute
an approximation factor. For this we need the optimal MIS sizes we have computed
for most of the graphs with the help of Gurobi beforehand.  This script collects
these optima, which are included in the files storing the graphs themselves.
"""

import argparse
import pathlib
import pandas as pd
import networkx as nx
import dgl
import torch
import random as rd
from tqdm.auto import tqdm

def main(args):
    rows = []

    for graph_file in tqdm(list(args.experiment_input_folder.rglob("test/*.gpickle"))):
        graph_name = graph_file.stem
        _G = nx.read_gpickle(graph_file)
        labels_given = "label" in rd.sample(_G.nodes(data=True), 1)[0][1].keys()
        weights_given = "weight" in rd.sample(_G.nodes(data=True), 1)[0][1].keys()
        
        node_attrs = []
        if labels_given:
            node_attrs += ["label"]
        if weights_given:
            node_attrs += ["weight"]

        G = dgl.from_networkx(_G, node_attrs=node_attrs)

        optimal_mis = None
        optimal_mwis = None

        if labels_given:
            _lbls = G.ndata["label"].detach()
            optimal_mis = torch.sum(_lbls).item()

            if weights_given:
                optimal_mwis = torch.sum(G.ndata['weight'][_lbls == 1]).item()

        weighted = "-rgw" in str(graph_file)
        rows.append((graph_name + "_weighted" if weighted else graph_name, optimal_mis, optimal_mwis))

    df = pd.DataFrame(rows, columns=["graph", "optimal_mis", "optimal_mwis"])
    df.to_csv(args.aggregation_output, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parser for MIS test graph pseudo-optima.")

    parser.add_argument("experiment_input_folder", type=pathlib.Path, action="store",  help="Folder in which the experiment input is given.")
    parser.add_argument("aggregation_output", type=pathlib.Path, action="store",  help="File into which to write the aggregated results as CSV.")

    args = parser.parse_args()
    args.aggregation_output.parent.mkdir(parents=True, exist_ok=True)

    main(args)
