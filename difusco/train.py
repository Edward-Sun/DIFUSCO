"""The handler for training and evaluation."""

import os
from argparse import ArgumentParser

import torch
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.utilities import rank_zero_info

from pl_tsp_model import TSPModel
from pl_mis_model import MISModel


def arg_parser():
  parser = ArgumentParser(description='Train a Pytorch-Lightning diffusion model on a TSP dataset.')
  parser.add_argument('--task', type=str, required=True)
  parser.add_argument('--storage_path', type=str, required=True)
  parser.add_argument('--training_split', type=str, default='data/tsp/tsp50_train_concorde.txt')
  parser.add_argument('--training_split_label_dir', type=str, default=None,
                      help="Directory containing labels for training split (used for MIS).")
  parser.add_argument('--validation_split', type=str, default='data/tsp/tsp50_test_concorde.txt')
  parser.add_argument('--test_split', type=str, default='data/tsp/tsp50_test_concorde.txt')
  parser.add_argument('--validation_examples', type=int, default=64)

  parser.add_argument('--batch_size', type=int, default=64)
  parser.add_argument('--num_epochs', type=int, default=50)
  parser.add_argument('--learning_rate', type=float, default=1e-4)
  parser.add_argument('--weight_decay', type=float, default=0.0)
  parser.add_argument('--lr_scheduler', type=str, default='constant')

  parser.add_argument('--num_workers', type=int, default=16)
  parser.add_argument('--fp16', action='store_true')
  parser.add_argument('--use_activation_checkpoint', action='store_true')

  parser.add_argument('--diffusion_type', type=str, default='gaussian')
  parser.add_argument('--diffusion_schedule', type=str, default='linear')
  parser.add_argument('--diffusion_steps', type=int, default=1000)
  parser.add_argument('--inference_diffusion_steps', type=int, default=1000)
  parser.add_argument('--inference_schedule', type=str, default='linear')
  parser.add_argument('--inference_trick', type=str, default="ddim")
  parser.add_argument('--sequential_sampling', type=int, default=1)
  parser.add_argument('--parallel_sampling', type=int, default=1)

  parser.add_argument('--n_layers', type=int, default=12)
  parser.add_argument('--hidden_dim', type=int, default=256)
  parser.add_argument('--sparse_factor', type=int, default=-1)
  parser.add_argument('--aggregation', type=str, default='sum')
  parser.add_argument('--two_opt_iterations', type=int, default=1000)
  parser.add_argument('--save_numpy_heatmap', action='store_true')

  parser.add_argument('--project_name', type=str, default='tsp_diffusion')
  parser.add_argument('--wandb_entity', type=str, default=None)
  parser.add_argument('--wandb_logger_name', type=str, default=None)
  parser.add_argument("--resume_id", type=str, default=None, help="Resume training on wandb.")
  parser.add_argument('--ckpt_path', type=str, default=None)
  parser.add_argument('--resume_weight_only', action='store_true')

  parser.add_argument('--do_train', action='store_true')
  parser.add_argument('--do_test', action='store_true')
  parser.add_argument('--do_valid_only', action='store_true')

  args = parser.parse_args()
  return args


def main(args):
  epochs = args.num_epochs
  project_name = args.project_name

  if args.task == 'tsp':
    model_class = TSPModel
    saving_mode = 'min'
  elif args.task == 'mis':
    model_class = MISModel
    saving_mode = 'max'
  else:
    raise NotImplementedError

  model = model_class(param_args=args)

  wandb_id = os.getenv("WANDB_RUN_ID") or wandb.util.generate_id()
  wandb_logger = WandbLogger(
      name=args.wandb_logger_name,
      project=project_name,
      entity=args.wandb_entity,
      save_dir=os.path.join(args.storage_path, f'models'),
      id=args.resume_id or wandb_id,
  )
  rank_zero_info(f"Logging to {wandb_logger.save_dir}/{wandb_logger.name}/{wandb_logger.version}")

  checkpoint_callback = ModelCheckpoint(
      monitor='val/solved_cost', mode=saving_mode,
      save_top_k=3, save_last=True,
      dirpath=os.path.join(wandb_logger.save_dir,
                           args.wandb_logger_name,
                           wandb_logger._id,
                           'checkpoints'),
  )
  lr_callback = LearningRateMonitor(logging_interval='step')

  trainer = Trainer(
      accelerator="auto",
      devices=torch.cuda.device_count() if torch.cuda.is_available() else None,
      max_epochs=epochs,
      callbacks=[TQDMProgressBar(refresh_rate=20), checkpoint_callback, lr_callback],
      logger=wandb_logger,
      check_val_every_n_epoch=1,
      strategy=DDPStrategy(static_graph=True),
      precision=16 if args.fp16 else 32,
  )

  rank_zero_info(
      f"{'-' * 100}\n"
      f"{str(model.model)}\n"
      f"{'-' * 100}\n"
  )

  ckpt_path = args.ckpt_path

  if args.do_train:
    if args.resume_weight_only:
      model = model_class.load_from_checkpoint(ckpt_path, param_args=args)
      trainer.fit(model)
    else:
      trainer.fit(model, ckpt_path=ckpt_path)

    if args.do_test:
      trainer.test(ckpt_path=checkpoint_callback.best_model_path)

  elif args.do_test:
    trainer.validate(model, ckpt_path=ckpt_path)
    if not args.do_valid_only:
      trainer.test(model, ckpt_path=ckpt_path)
  trainer.logger.finalize("success")


if __name__ == '__main__':
  args = arg_parser()
  main(args)
