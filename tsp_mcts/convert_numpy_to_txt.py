import os
import fire
import numpy as np


def main(
    heatmap_dir,
    output_dir,
    num_nodes=10000,
    num_files=16,
    expected_valid_prob=0.02,
    heatmap_prefix="heatmap",
):
    for i in range(num_files):
        file_name = f"{heatmap_dir}/numpy_heatmap/test-{heatmap_prefix}-{i}.npy"
        print(file_name)
        points_file_name = f"{heatmap_dir}/numpy_heatmap/test-points-{i}.npy"
        adj_matrix = np.load(file_name)
        points = np.load(points_file_name)

        dists = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
        adj_matrix = adj_matrix + 0.01 * (1.0 - dists)

        adj_matrix[adj_matrix == np.inf] = 0.0

        expected_valid_value_num = int(num_nodes * num_nodes * expected_valid_prob)
        valid_values = adj_matrix[(adj_matrix > 0.0)]
        valid_values = np.sort(valid_values)
        valid_value_threshold = valid_values[-expected_valid_value_num]
        print("valid_value_threshold", valid_value_threshold)
        print("prenorm_max", adj_matrix.max())
        top3_nodes_per_node = np.argsort(adj_matrix, axis=1)[:, -3:]

        valid_mask = adj_matrix > valid_value_threshold
        # top3_mask = np.zeros_like(adj_matrix, dtype=np.bool)
        # for k in range(num_nodes):
        #   top3_mask[k, top3_nodes_per_node[k]] = True
        # fast top3 mask

        top3_mask = np.zeros_like(adj_matrix, dtype=np.bool)
        top3_mask[np.arange(num_nodes)[:, None], top3_nodes_per_node] = True

        valid_mask = valid_mask | top3_mask
        adj_matrix = adj_matrix * valid_mask
        adj_matrix[adj_matrix != 0.0] += 1e-2
        adj_matrix = adj_matrix + adj_matrix.T
        adj_matrix = adj_matrix / adj_matrix.sum(axis=1, keepdims=True)

        print("valid_prob", (adj_matrix > 0.0).mean())
        print("valid_mean", adj_matrix[(adj_matrix > 0.0)].mean())
        print("valid_std", adj_matrix[(adj_matrix > 0.0)].std())
        print("valid_min", adj_matrix[(adj_matrix > 0.0)].min())
        print("valid_max", adj_matrix[(adj_matrix > 0.0)].max())
        print("min_valid_prob_per_node", (adj_matrix > 0.0).sum(axis=1).min())
        print("max_valid_prob_per_node", (adj_matrix > 0.0).sum(axis=1).max())

        # Output the normalized results
        # first line is num_nodes
        # each line is a row of the adjacency matrix with 6 digits after the decimal point
        folder_name = f"{output_dir}/{heatmap_prefix}/tsp{num_nodes}"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        file_name = f"{folder_name}/heatmaptsp{num_nodes}_{i}.txt"
        with open(file_name, "w") as f:
            output_strings = []
            output_strings.append(f"{num_nodes}\n")
            for row in range(num_nodes):
                output_strings.append(
                    " ".join([f"{x:.6f}" for x in adj_matrix[row]]) + "\n"
                )
            f.write("".join(output_strings))
        print(file_name)


if __name__ == "__main__":
    fire.Fire(main)
