# Independent Set Benchmarking Suite

This repository contains a benchmarking suite for maximum independent set
solvers. The suite bundles several solvers and datasets with a unified
interface. Currently, we support the following solvers:

- Gurobi
- Intel-TreeSearch
- DGL-TreeSearch
- KaMIS
- Learning What To Defer

## Details on the Solvers

### Gurobi
Gurobi is a commercial mathematical optimization solver. There are various ways
of formulating the Maximum (Weighted) Independent Set problem mathematically. By
default, we use a linear program, and access Gurobi using PuLP. We also support
a quadratic formulation. For details, we refer to our paper (see below).

### Intel-TreeSearch
The Intel-TreeSearch is the algorithm proposed by [Li et
al.](https://arxiv.org/pdf/1810.10659.pdf). The idea is to train a graph
convolutional network (GCN), which assigns each vertex a probability of
belonging to the independent set, and then greedily and iteratively assign
vertices to the set. They furthermore employ the reduction and local search
algorithms by KaMIS to speed up the computation. We use their [published
code](https://github.com/isl-org/NPHard), which unfortunately is not runnable in
its default state. We apply a git patch to make the code runnable, enable
further evaluation by collecting statistics, and add command-line flags for more
fine-grained control of the solver configuration.

### DGL-TreeSearch
Because the code provided by Li et al. might be difficult to read and maintain,
and hence is prone to errors in the evaluation, we re-implement the tree search
using PyTorch and the established Deep Graph Library. Our implementation aims at
offering a more readable and modern implementation, which benefits from
improvements in the two deep learning libraries during recent years.
Furthermore, it fixes various issues of the original implementation that
sometimes deviates from the paper. Additionally, we implement further techniques
to improve the search, like queue pruning, and weighted selection of the next
element, as well as multi-GPU functionality.

### KaMIS
KaMIS is an open-source solver tailored towards the MIS and MWIS problems. It
offers support both for the unweighted case as well as the weighted case. It
employs graph kernelization and an optimized branch-and-bound algorithm to
efficiently find independent sets. Note that the algorithms and techniques
differ between the weighted and unweighted cases. We use the code unmodified
from the [official repository](https://github.com/KarlsruheMIS/KaMIS) and
integrate it within the suite.

### Learning what to Defer
Learning what to Defer (LwD) is an unsupervised deep reinforcement
learning-based solution introduced by [Ahn et
al.](https://arxiv.org/abs/2006.09607). Their idea is similar to the tree
search, as the algorithm iteratively assigns vertices to the independent set.
However, this is not done using a supervised GCN, but instead by an unsupervised
agent built upon the GraphSAGE architecture and trained by Proximal Policy
Optimization. There is no queue of partial solutions.  As [their
code](https://github.com/sungsoo-ahn/learning_what_to_defer) does not work with
generic input, we patch it.

## Repository Contents

In `solvers`, you can find the wrappers for the currently supported solvers. In `data_generation`, you find the code required for generating random and real-world graphs.

For using this suite, `conda` is required. You can the `setup_bm_env.sh` script which will setup the conda environment with all required dependencies. You can find out more about the usage using `python main.py -h`. The `main.py` file is the main interface you will call for data generation, solving, and training.

In the `helper_scripts` folder, you find some scripts that could be helpful when doing analyses with this suite.

## Publication

You can find our ICLR 2022 conference paper [here](https://openreview.net/forum?id=mk0HzdqY7i1).

If you use this in your work, please cite us and the papers of the solvers that you use.

```bibtex
@inproceedings{boether_dltreesearch_2022,
  author = {Böther, Maximilian and Kißig, Otto and Taraz, Martin and Cohen, Sarel and Seidel, Karen and Friedrich, Tobias},
  title = {What{\textquoteright}s Wrong with Deep Learning in Tree Search for Combinatorial Optimization},
  booktitle = {Proceedings of the International Conference on Learning Representations ({ICLR})},
  year = {2022}
}
```

If you have questions you are welcome to reach out to [@MaxiBoether](https://github.com/MaxiBoether) and [@EightSQ](https://github.com/EightSQ).

### Data and Models

On popular request, we provide the (small) random graphs with labels and the models we trained [here](https://owncloud.hpi.de/s/cv6szEJtSs8UGju) ([backup location](https://mboether.com/paper-models-randomgraphs.zip)).
The Intel tree search model that was trained by Li et al. can be downloaded from [the original repository](https://github.com/isl-org/NPHard/tree/master/model).
Note that we cannot reupload the labeled real world graphs, as we do not have any permission to re-distribute them.
However, the benchmarking suite supports the automatic download and labeling of _all_ random and real world graphs used in the paper.
Please do not only rely on the data we provide and instead use this suite to generate graphs and train models on your own, as there is no guarantee that our evaluation is fully correct.

## Contributions

There are (of course) some improvements that can be made. For example, the argument parsing requires a major refactoring, and the output formats are currently not fully harmonized. We are open for pull requests, if you want to contribute. Thank you very much!
