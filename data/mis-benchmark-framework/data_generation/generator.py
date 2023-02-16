from abc import ABC, abstractmethod, abstractstaticmethod
from pathlib import Path
import os
import shutil
import pickle
from solvers.gurobi import Gurobi
import json
import numpy as np
from logzero import logger
import multiprocessing


class DataGenerator(ABC):
  def _call_gurobi_solver(self, G, timeout=30, weighted=False, use_multiprocessing=False):
    if self.output_path is None:
      raise ValueError("This function can only be called if an output path is set!")

    # create temp directories
    tmp_input_folder = self.output_path / "gurobi_input"
    tmp_output_folder = self.output_path / "gurobi_output"

    try:
      os.mkdir(tmp_input_folder)
    except:
      pass
    try:
      os.mkdir(tmp_output_folder)
    except:
      pass

    if use_multiprocessing:
      tmp_input_folder = tmp_input_folder / str(multiprocessing.current_process().name)
      tmp_output_folder = tmp_input_folder / str(multiprocessing.current_process().name)

    #         if tmp_input_folder.exists() and tmp_input_folder.is_dir():
    #             shutil.rmtree(tmp_input_folder)
    #         if tmp_output_folder.exists() and tmp_output_folder.is_dir():
    #             shutil.rmtree(tmp_output_folder)

    try:
      os.mkdir(tmp_input_folder)
      os.mkdir(tmp_output_folder)
    except:
      pass

    # write input file for gurobi
    input_file = tmp_input_folder / "input.gpickle"
    with open(input_file, "wb") as f:
      pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)

    # call gurobi
    solver = Gurobi()
    parameters = {"time_limit": timeout, "loglevel": "INFO", "num_threads": 16}

    if weighted:
      parameters["weighted"] = "yes"

    solver.solve(tmp_input_folder, tmp_output_folder, parameters)
    raise NotImplementedError

    # read output from gurobi
    with open(tmp_output_folder / "results.json") as f:
      results = json.load(f)

    mis = results["input"]["mwis"]
    status = results["input"]["gurobi_status"]

    # cleanup
    # shutil.rmtree(tmp_input_folder)
    # shutil.rmtree(tmp_output_folder)

    return mis, status

  def random_weight(self, n, mu=1, sigma=0.1):
    return np.around(np.random.normal(mu, sigma, n)).astype(int).clip(min=0)

  @abstractmethod
  def generate(self, gen_labels=False, weighted=False):
    pass
