#!/usr/bin/env python3

"""
If you run experiments with different solvers on different graphs, you probably want to
have a single csv file containing all data. This script takes the experiment output folder
as input, and outputs such an aggregated csv.
"""

import argparse
import pathlib
import re
import pandas as pd
import json


def parse_path(results_json_path: str, outputs_path: str) -> re.Match:
    # output path format
    # /(solver)/(pathto/to/graphtype)/test/results.json
    results_json_path = results_json_path.replace(outputs_path, "")
    match = re.match(r"\/(?P<solver>\S+?)\/(?P<dataset>\S+)\/test(?:\/(?P<config>\S+))?\/results\.json", results_json_path)
    if not match:
        raise Exception(f"Could not parse path {results_json_path}.")
    return match

def _parse_kamis_results(doc):
    metrics = {}
    metrics["found"] = doc.get("found_mis", doc.get("mwis_found"))
    if metrics["found"]:
        metrics["solution_time"] = doc.get("solution_time", doc.get("time_to_find_mwis"))
        metrics["solution_size"] = doc.get("vertices", doc.get("mwis_weight"))
    return metrics

def _parse_dgl_results(doc):
    metrics = {}
    metrics["found"] = doc.get("mwis_found")
    if metrics["found"]:
        metrics["solution_time"] = doc.get("time_to_find_mwis")
        metrics["process_time"] = doc.get("process_time_to_find_mwis")
        metrics["solution_size"] = doc.get("mwis_weight")
    return metrics

def _parse_intel_results(doc):
    metrics = {}
    metrics["found"] = doc.get("found_mis")
    if metrics["found"]:
        metrics["solution_time"] = doc.get("solution_time")
        metrics["process_time"] = doc.get("solution_process_time")
        metrics["solution_size"] = doc.get("vertices")
    metrics["total_time"] = doc.get("total_time")
    return metrics

def _parse_gurobi_results(doc):
    metrics = {}
    metrics["found"] = True # This is the way it is
    metrics["solution_time"] = doc.get("gurobi_explore_time")
    metrics["total_time"] = doc.get("total_time")
    metrics["solution_size"] = doc.get("mwis_weight")
    return metrics

def _parse_lwd_results(doc):
    metrics = {}
    metrics["found"] = doc.get("found_mis")
    if metrics["found"]:
        metrics["solution_time"] = doc.get("solution_time")
        metrics["process_time"] = doc.get("solution_process_time")
        metrics["solution_size"] = doc.get("vertices")
    metrics["total_time"] = doc.get("total_time")
    return metrics

def parse_output_json(results_json_path, solver):
    if results_json_path.stat().st_size == 0: # empty results.json, i.e., timeouted experiment                                                                                                                                                                                                                               
        yield None, None
    else:
        with open(results_json_path, "r") as f:
            doc = json.load(f)
        for graph in doc.keys():
            if solver == "kamis":
                yield graph, _parse_kamis_results(doc[graph])
            elif solver == "dgl-treesearch":
                yield graph, _parse_dgl_results(doc[graph])
            elif solver == "intel-treesearch":
                yield graph, _parse_intel_results(doc[graph]) or {}
            elif solver == "gurobi":
                yield graph, _parse_gurobi_results(doc[graph])
            elif solver == "lwd":
                yield graph, _parse_lwd_results(doc[graph])

def main(args):
    rows = []
    metric_names = [
        "found",
        "solution_time",
        "solution_size",
        "total_time"
    ]
    for result_file in args.experiment_output_folder.rglob("results.json"):
        m = parse_path(str(result_file), str(args.experiment_output_folder))
        solver = m["solver"]
        config = m["config"]
        weighted = "-rgw" in str(result_file)
        for graph, metrics in parse_output_json(result_file, solver):
            if graph is not None and metrics is not None:
                rows.append((solver, config, graph + "_weighted" if weighted else graph) + tuple(metrics.get(metric_name) for metric_name in metric_names))

    df = pd.DataFrame(rows, columns=(["solver", "config", "graph"] + metric_names))
    df.to_csv(args.aggregation_output, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregator for the testbench results.")

    parser.add_argument("experiment_output_folder", type=pathlib.Path, action="store",  help="Folder in which the output by the experiments are stored.")
    parser.add_argument("aggregation_output", type=pathlib.Path, action="store",  help="File into which to write the aggregated results as CSV.")

    args = parser.parse_args()
    args.aggregation_output.parent.mkdir(parents=True, exist_ok=True)

    main(args)
