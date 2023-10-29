## MCTS inference for TSP heatmaps

The code is adopted from the CPU version of [Spider-scnu/TSP](https://github.com/Spider-scnu/TSP/tree/master/MCTS-CPUver).

Please place `tsp500_test_concorde.txt`, `tsp1000_test_concorde.txt`, and `tsp10000_test_concorde.txt` under the current direcotry. They can be downloaded from [the original repo](https://github.com/Spider-scnu/TSP/tree/master#dataset).

### Usage

First, run the evaluation script for the diffusion model with `--save_numpy_heatmap True`.

Next, run the MCTS code:
```bash
# You need to modify some hard-coded configuration in the cpp code following the instructions in the script before running it.
bash solve-500.sh
bash solve-1000.sh
bash solve-10000.sh
```
