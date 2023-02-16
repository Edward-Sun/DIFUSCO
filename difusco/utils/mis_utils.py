import numpy as np

def mis_decode_np(predictions, adj_matrix):
  """Decode the labels to the MIS."""
  solution = np.zeros_like(predictions.astype(int))
  sorted_predict_labels = np.argsort(- predictions)
  csr_adj_matrix = adj_matrix.tocsr()

  for i in sorted_predict_labels:
    next_node = i

    if solution[next_node] == -1:
      continue

    solution[csr_adj_matrix[next_node].nonzero()[1]] = -1
    solution[next_node] = 1

  return (solution == 1).astype(int)
