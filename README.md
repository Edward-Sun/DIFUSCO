# DIFUSCO

See ["DIFUSCO: Graph-based Diffusion Solvers for Combinatorial Optimization"](TBD) for the paper associated with this codebase.

## Setup

```bash
conda env create -f environment.yml
conda activate difusco
```

## Codebase Structure

* `difusco/pl_meta_model.py`: the code for a meta pytorch-lightning model for training and evaluation.
* `difusco/pl_tsp_model.py`: the code for the TSP problem
* `difusco/pl_mis_model.py`: the code for the MIS problem
* `difusco/trian.py`: the handler for training and evaluation

## Data

Please check the `data` folder.

## Reproduction

Please check the [reproducing_scripts](reproducing_scripts.md) for more details.

## Pretrained Checkpoints

Please download the pretrained model checkpoints from [here](https://drive.google.com/drive/folders/1IjaWtkqTAs7lwtFZ24lTRspE0h1N6sBH?usp=sharing).
