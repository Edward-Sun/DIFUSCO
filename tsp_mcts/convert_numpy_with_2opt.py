import os
import torch
import time
import numpy as np


from tsp_utils import TSP_2opt
from cython_merge.cython_merge import merge_cython


def cython_merge(points, adj_mat):
  real_adj_mat, merge_iterations = merge_cython(points.astype("double"), adj_mat.astype("double"))
  real_adj_mat = np.asarray(real_adj_mat)
  return real_adj_mat, merge_iterations


def two_opt_torch(points, tour, max_iterations=1000, device="cpu"):
  iterator = 0
  tour = tour.copy()
  with torch.inference_mode():
    cuda_points = torch.from_numpy(points).to(device)
    cuda_tour = torch.from_numpy(tour).to(device)
    min_change = -1.0
    while min_change < 0.0:
      points_i = cuda_points[cuda_tour[:-1]].reshape((-1, 1, 2))
      points_j = cuda_points[cuda_tour[:-1]].reshape((1, -1, 2))
      points_i_plus_1 = cuda_points[cuda_tour[1:]].reshape((-1, 1, 2))
      points_j_plus_1 = cuda_points[cuda_tour[1:]].reshape((1, -1, 2))

      A_ij = torch.sqrt(torch.sum((points_i - points_j) ** 2, axis=-1))
      A_i_plus_1_j_plus_1 = torch.sqrt(torch.sum((points_i_plus_1 - points_j_plus_1) ** 2, axis=-1))
      A_i_i_plus_1 = torch.sqrt(torch.sum((points_i - points_i_plus_1) ** 2, axis=-1))
      A_j_j_plus_1 = torch.sqrt(torch.sum((points_j - points_j_plus_1) ** 2, axis=-1))

      change = A_ij + A_i_plus_1_j_plus_1 - A_i_i_plus_1 - A_j_j_plus_1
      valid_change = torch.triu(change, diagonal=2)

      min_change = torch.min(valid_change)
      flatten_argmin_index = torch.argmin(valid_change)
      min_i = torch.div(flatten_argmin_index, len(points), rounding_mode='floor')
      min_j = torch.remainder(flatten_argmin_index, len(points))

      if min_change < -1e-6:
        cuda_tour[min_i + 1:min_j + 1] = torch.flip(cuda_tour[min_i + 1:min_j + 1], dims=(0,))
        iterator += 1
      else:
        break

      if iterator >= max_iterations:
        break
    tour = cuda_tour.cpu().numpy()
  return tour, iterator


def main(
    heatmap_dir,
    num_files=16,
    heatmap_prefix="heatmap",
):
    for i in range(num_files):
        file_name = f"{heatmap_dir}/numpy_heatmap/test-{heatmap_prefix}-{i}.npy"
        print(file_name)
        points_file_name = f"{heatmap_dir}/numpy_heatmap/test-points-{i}.npy"
      adj_mat = np.load(file_name)
      points = np.load(points_file_name)

      union_find_t1 = time.time()
      # real_adj_mat, merge_iterations = numpy_merge(points, adj_mat)
      real_adj_mat, merge_iterations = cython_merge(points, adj_mat)

      union_find_t2 = time.time()
      print(f'Union find time: {union_find_t2 - union_find_t1}')

      tour = [0]
      while len(tour) < adj_mat.shape[0] + 1:
        n = np.nonzero(real_adj_mat[tour[-1]])[0]
        if len(tour) > 1:
          n = n[n != tour[-2]]
        tour.append(n.max())

      # Refine using 2-opt
      tsp_solver = TSP_2opt(points)
      original_tour = tour
      device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      solved_tour, ns = two_opt_torch(points.astype("float64"), np.array(tour).astype('int64'),
                                      max_iterations=5000, device=device)

      def has_duplicates(l):
        existing = set()
        for item in l:
          if item in existing:
            return True
          existing.add(item)
        return False

      assert solved_tour[-1] == solved_tour[0], 'Tour not a cycle'
      assert not has_duplicates(solved_tour[:-1]), 'Tour not Hamiltonian'
      opt_t3 = time.time()
      print(f'2-opt time: {opt_t3 - union_find_t2}, {ns} iterations')

      # convert tour back to adj matrix with numpy operations
      solved_adj_mat = np.zeros_like(adj_mat)
      solved_adj_mat[solved_tour[:-1], solved_tour[1:]] = 1.0

      np.save(f"{heatmap_dir}/numpy_heatmap/test-2opt-{heatmap_prefix}-{i}.npy",
              solved_adj_mat)

      solved_cost = tsp_solver.evaluate(solved_tour)
      solved_costs.append(solved_cost)
      print(f'Cost: {solved_cost}')

    print("mean_solved_costs", np.mean(solved_costs))


if __name__ == "__main__":
    fire.Fire(main)
