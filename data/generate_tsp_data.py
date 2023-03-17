import argparse
import pprint as pp
import time
import warnings
from multiprocessing import Pool

import lkh
import numpy as np
import tqdm
import tsplib95
from concorde.tsp import TSPSolver  # https://github.com/jvkersch/pyconcorde

warnings.filterwarnings("ignore")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--min_nodes", type=int, default=20)
  parser.add_argument("--max_nodes", type=int, default=50)
  parser.add_argument("--num_samples", type=int, default=128000)
  parser.add_argument("--batch_size", type=int, default=128)
  parser.add_argument("--filename", type=str, default=None)
  parser.add_argument("--solver", type=str, default="lkh")
  parser.add_argument("--lkh_trails", type=int, default=1000)
  parser.add_argument("--seed", type=int, default=1234)
  opts = parser.parse_args()

  assert opts.num_samples % opts.batch_size == 0, "Number of samples must be divisible by batch size"

  np.random.seed(opts.seed)

  if opts.filename is None:
    opts.filename = f"tsp{opts.min_nodes}-{opts.max_nodes}_concorde.txt"

  # Pretty print the run args
  pp.pprint(vars(opts))

  with open(opts.filename, "w") as f:
    start_time = time.time()
    for b_idx in tqdm.tqdm(range(opts.num_samples // opts.batch_size)):
      num_nodes = np.random.randint(low=opts.min_nodes, high=opts.max_nodes + 1)
      assert opts.min_nodes <= num_nodes <= opts.max_nodes

      batch_nodes_coord = np.random.random([opts.batch_size, num_nodes, 2])

      def solve_tsp(nodes_coord):
        if opts.solver == "concorde":
          scale = 1e6
          solver = TSPSolver.from_data(nodes_coord[:, 0] * scale, nodes_coord[:, 1] * scale, norm="EUC_2D")
          solution = solver.solve(verbose=False)
          tour = solution.tour
        elif opts.solver == "lkh":
          scale = 1e6
          lkh_path = 'LKH-3.0.6/LKH'
          problem = tsplib95.models.StandardProblem()
          problem.name = 'TSP'
          problem.type = 'TSP'
          problem.dimension = num_nodes
          problem.edge_weight_type = 'EUC_2D'
          problem.node_coords = {n + 1: nodes_coord[n] * scale for n in range(num_nodes)}

          solution = lkh.solve(lkh_path, problem=problem, max_trials=opts.lkh_trails, runs=10)
          tour = [n - 1 for n in solution[0]]
        else:
          raise ValueError(f"Unknown solver: {opts.solver}")

        return tour

      with Pool(opts.batch_size) as p:
        tours = p.map(solve_tsp, [batch_nodes_coord[idx] for idx in range(opts.batch_size)])

      for idx, tour in enumerate(tours):
        if (np.sort(tour) == np.arange(num_nodes)).all():
          f.write(" ".join(str(x) + str(" ") + str(y) for x, y in batch_nodes_coord[idx]))
          f.write(str(" ") + str('output') + str(" "))
          f.write(str(" ").join(str(node_idx + 1) for node_idx in tour))
          f.write(str(" ") + str(tour[0] + 1) + str(" "))
          f.write("\n")

    end_time = time.time() - start_time

    assert b_idx == opts.num_samples // opts.batch_size - 1

  print(f"Completed generation of {opts.num_samples} samples of TSP{opts.min_nodes}-{opts.max_nodes}.")
  print(f"Total time: {end_time / 60:.1f}m")
  print(f"Average time: {end_time / opts.num_samples:.1f}s")
