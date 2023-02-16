# Reproduce Results

## Training

### Training on TSP50

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# shellcheck disable=SC2155
export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
echo "WANDB_ID is $WANDB_RUN_ID"

python -u difusco/train.py \
  --task "tsp" \
  --wandb_logger_name "tsp_diffusion_graph_categorical_tsp50" \
  --diffusion_type "categorical" \
  --do_train \
  --do_test \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "/your/storage/path" \
  --training_split "/your/tsp50_train_concorde.txt" \
  --validation_split "/your/tsp50_valid_concorde.txt" \
  --test_split "/your/tsp50_test_concorde.txt" \
  --batch_size 64 \
  --num_epochs 50 \
  --validation_examples 8 \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 50
```

### Training on TSP100

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# shellcheck disable=SC2155
export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
echo "WANDB_ID is $WANDB_RUN_ID"

python -u difusco/train.py \
  --task "tsp" \
  --wandb_logger_name "tsp_diffusion_graph_categorical_tsp100" \
  --diffusion_type "categorical" \
  --do_train \
  --do_test \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "/your/storage/path" \
    --training_split "/your/tsp100_train_concorde.txt" \
    --validation_split "/your/tsp100_valid_concorde.txt" \
    --test_split "/your/tsp100_test_concorde.txt" \
  --batch_size 32 \
  --num_epochs 50 \
  --validation_examples 8 \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 50
```

### Training on TSP500

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# shellcheck disable=SC2155
export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
echo "WANDB_ID is $WANDB_RUN_ID"

python -u difusco/train.py \
  --task "tsp" \
  --wandb_logger_name "tsp_diffusion_graph_categorical_tsp500" \
  --diffusion_type "categorical" \
  --do_train \
  --do_test \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "/your/storage/path" \
  --training_split "/your/tsp500_train_concorde.txt" \
  --validation_split "/your/tsp500_valid_concorde.txt" \
  --test_split "/your/tsp500_test_concorde.txt" \
  --sparse_factor 50 \
  --batch_size 8 \
  --num_epochs 50 \
  --validation_examples 8 \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 50
```

### Training on TSP1000

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# shellcheck disable=SC2155
export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
echo "WANDB_ID is $WANDB_RUN_ID"

python -u difusco/train.py \
  --task "tsp" \
  --wandb_logger_name "tsp_diffusion_graph_categorical_tsp500" \
  --diffusion_type "categorical" \
  --do_train \
  --do_test \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "/your/storage/path" \
  --training_split "/your/tsp1000_train_concorde.txt" \
  --validation_split "/your/tsp1000_valid_concorde.txt" \
  --test_split "/your/tsp1000_test_concorde.txt" \
  --sparse_factor 100 \
  --batch_size 8 \
  --num_epochs 50 \
  --validation_examples 8 \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 50 \
  --use_activation_checkpoint
```

### Training on TSP10000

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# shellcheck disable=SC2155
export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
echo "WANDB_ID is $WANDB_RUN_ID"

python -u difusco/train.py \
  --task "tsp" \
  --wandb_logger_name "tsp_diffusion_graph_categorical_tsp500" \
  --diffusion_type "categorical" \
  --do_train \
  --do_test \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "/your/storage/path" \
  --training_split "/your/tsp10000_train_concorde.txt" \
  --validation_split "/your/tsp10000_valid_concorde.txt" \
  --test_split "/your/tsp10000_test_concorde.txt" \
  --sparse_factor 100 \
  --batch_size 1 \
  --num_epochs 50 \
  --validation_examples 8 \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 50 \
  --two_opt_iterations 5000 \
  --use_activation_checkpoint
```

### Training on SATLIB graphs of the MIS problem

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# shellcheck disable=SC2155
export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
echo "WANDB_ID is $WANDB_RUN_ID"

python -u difusco/train.py \
  --task "mis" \
  --wandb_logger_name "mis_diffusion_graph_categorical_sat" \
  --diffusion_type "categorical" \
  --do_train \
  --do_test \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "/your/storage/path" \
  --training_split "/your/train_mis_sat/*gpickle" \
  --validation_split "/your/test_mis_sat/*gpickle" \
  --test_split "/your/test_mis_sat/*gpickle" \
  --batch_size 16 \
  --num_epochs 50 \
  --validation_examples 8 \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 50 \
  --use_activation_checkpoint
```

### Training on ER-[700-800] graphs of the MIS problem

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# shellcheck disable=SC2155
export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
echo "WANDB_ID is $WANDB_RUN_ID"

python -u difusco/train.py \
  --task "mis" \
  --wandb_logger_name "mis_diffusion_graph_gaussian_er" \
  --diffusion_type "gaussian" \
  --do_train \
  --do_test \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "/your/storage/path" \
  --training_split "/your/data_er/train/*gpickle" \
  --training_split_label_dir "/your/data_er/train_annotations/" \
  --validation_split "/your/data_er/train/validation/*gpickle" \
  --test_split "/your/data_er/train/test/*gpickle" \
  --batch_size 4 \
  --num_epochs 50 \
  --validation_examples 8 \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 50 \
  --use_activation_checkpoint
```

## Evaluation

Due to the various possible configurations for evaluation, we only provide a few examples of evaluation commands.

### Evaluation on TSP100-Categorical with greedy decoding

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -u difusco/train.py \
  --task "tsp" \
  --wandb_logger_name "tsp_diffusion_graph_categorical_tsp100_test" \
  --diffusion_type "categorical" \
  --do_test \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "/your/storage/path" \
  --training_split "/your/tsp100_train_concorde.txt" \
  --validation_split "/your/tsp100_valid_concorde.txt" \
  --test_split "/your/tsp100_test_concorde.txt" \
  --batch_size 32 \
  --num_epochs 25 \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 50 \
  --ckpt_path "/your/tsp100_categorical/ckpt_path/last.ckpt" \
  --resume_weight_only
```

### Evaluation on TSP500-Categorical with sampling decoding (4x parallel)

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -u difusco/train.py \
  --task "tsp" \
  --wandb_logger_name "tsp_diffusion_graph_gaussian_tsp500_test_parallel4" \
  --diffusion_type "categorical" \
  --do_test \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "/your/storage/path" \
  --training_split "/your/tsp500_train_concorde.txt" \
  --validation_split "/your/tsp500_valid_concorde.txt" \
  --test_split "/your/tsp500_test_concorde.txt" \
  --sparse_factor 50 \
  --batch_size 32 \
  --num_epochs 25 \
  --validation_examples 8 \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 50 \
  --parallel_sampling 4 \
  --ckpt_path "/your/tsp500_categorical/ckpt_path/last.ckpt" \
  --resume_weight_only
```

### Evaluation on TSP10000-Categorical with sampling decoding (4x sequential)

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -u difusco/train.py \
  --task "tsp" \
  --wandb_logger_name "tsp_diffusion_graph_gaussian_tsp10k_test_sequential4" \
  --diffusion_type "categorical" \
  --do_test \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "/your/storage/path" \
  --training_split "/your/tsp10000_train_concorde.txt" \
  --validation_split "/your/tsp10000_valid_concorde.txt" \
  --test_split "/your/tsp10000_test_concorde.txt" \
  --sparse_factor 100 \
  --batch_size 1 \
  --num_epochs 25 \
  --validation_examples 8 \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 50 \
  --sequential_sampling 4 \
  --two_opt_iterations 5000 \
  --ckpt_path "/your/tsp10k_categorical/ckpt_path/last.ckpt" \
  --resume_weight_only
```

### Evaluation on SATLIB graphs of the MIS problem with Categorical Diffusion models and greedy decoding

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -u difusco/train.py \
  --task "mis" \
  --wandb_logger_name "mis_diffusion_graph_categorical_sat_test" \
  --diffusion_type "categorical" \
  --do_test \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "/your/storage/path" \
  --training_split "/your/train_mis_sat/*gpickle" \
  --validation_split "/your/test_mis_sat/*gpickle" \
  --test_split "/your/test_mis_sat/*gpickle" \
  --batch_size 16 \
  --num_epochs 50 \
  --validation_examples 8 \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 50 \
  --ckpt_path "/your/mis_sat_categorical/ckpt_path/last.ckpt" \
  --resume_weight_only
```

### Evaluation on ER-[700-800] graphs of the MIS problem with Gaussian Diffusion models and sampling decoding (4x parallel)

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -u difusco/train.py \
  --task "mis" \
  --wandb_logger_name "mis_diffusion_graph_gaussian_er_test" \
  --diffusion_type "gaussian" \
  --do_test \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "/your/storage/path" \
  --training_split "/your/data_er/train/*gpickle" \
  --training_split_label_dir "/your/data_er/train_annotations/" \
  --validation_split "/your/data_er/train/validation/*gpickle" \
  --test_split "/your/data_er/train/test/*gpickle" \
  --batch_size 4 \
  --num_epochs 50 \
  --validation_examples 8 \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 50 \
  --parallel_sampling 4 \
  --use_activation_checkpoint \
  --ckpt_path "/your/mis_er_gaussian/ckpt_path/last.ckpt" \
  --resume_weight_only
```
