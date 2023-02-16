"""A meta PyTorch Lightning model for training and evaluating DIFUSCO models."""

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.utilities import rank_zero_info

from models.gnn_encoder import GNNEncoder
from utils.lr_schedulers import get_schedule_fn
from utils.diffusion_schedulers import CategoricalDiffusion, GaussianDiffusion


class COMetaModel(pl.LightningModule):
  def __init__(self,
               param_args,
               node_feature_only=False):
    super(COMetaModel, self).__init__()
    self.args = param_args
    self.diffusion_type = self.args.diffusion_type
    self.diffusion_schedule = self.args.diffusion_schedule
    self.diffusion_steps = self.args.diffusion_steps
    self.sparse = self.args.sparse_factor > 0 or node_feature_only

    if self.diffusion_type == 'gaussian':
      out_channels = 1
      self.diffusion = GaussianDiffusion(
          T=self.diffusion_steps, schedule=self.diffusion_schedule)
    elif self.diffusion_type == 'categorical':
      out_channels = 2
      self.diffusion = CategoricalDiffusion(
          T=self.diffusion_steps, schedule=self.diffusion_schedule)
    else:
      raise ValueError(f"Unknown diffusion type {self.diffusion_type}")

    self.model = GNNEncoder(
        n_layers=self.args.n_layers,
        hidden_dim=self.args.hidden_dim,
        out_channels=out_channels,
        aggregation=self.args.aggregation,
        sparse=self.sparse,
        use_activation_checkpoint=self.args.use_activation_checkpoint,
        node_feature_only=node_feature_only,
    )
    self.num_training_steps_cached = None

  def test_epoch_end(self, outputs):
    unmerged_metrics = {}
    for metrics in outputs:
      for k, v in metrics.items():
        if k not in unmerged_metrics:
          unmerged_metrics[k] = []
        unmerged_metrics[k].append(v)

    merged_metrics = {}
    for k, v in unmerged_metrics.items():
      merged_metrics[k] = float(np.mean(v))
    self.logger.log_metrics(merged_metrics, step=self.global_step)

  def get_total_num_training_steps(self) -> int:
    """Total training steps inferred from datamodule and devices."""
    if self.num_training_steps_cached is not None:
      return self.num_training_steps_cached
    dataset = self.train_dataloader()
    if self.trainer.max_steps and self.trainer.max_steps > 0:
      return self.trainer.max_steps

    dataset_size = (
        self.trainer.limit_train_batches * len(dataset)
        if self.trainer.limit_train_batches != 0
        else len(dataset)
    )

    num_devices = max(1, self.trainer.num_devices)
    effective_batch_size = self.trainer.accumulate_grad_batches * num_devices
    self.num_training_steps_cached = (dataset_size // effective_batch_size) * self.trainer.max_epochs
    return self.num_training_steps_cached

  def configure_optimizers(self):
    rank_zero_info('Parameters: %d' % sum([p.numel() for p in self.model.parameters()]))
    rank_zero_info('Training steps: %d' % self.get_total_num_training_steps())

    if self.args.lr_scheduler == "constant":
      return torch.optim.AdamW(
          self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)

    else:
      optimizer = torch.optim.AdamW(
          self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
      scheduler = get_schedule_fn(self.args.lr_scheduler, self.get_total_num_training_steps())(optimizer)

      return {
          "optimizer": optimizer,
          "lr_scheduler": {
              "scheduler": scheduler,
              "interval": "step",
          },
      }

  def categorical_posterior(self, target_t, t, x0_pred_prob, xt):
    """Sample from the categorical posterior for a given time step.
       See https://arxiv.org/pdf/2107.03006.pdf for details.
    """
    diffusion = self.diffusion

    if target_t is None:
      target_t = t - 1
    else:
      target_t = torch.from_numpy(target_t).view(1)

    if target_t > 0:
      Q_t = np.linalg.inv(diffusion.Q_bar[target_t]) @ diffusion.Q_bar[t]
      Q_t = torch.from_numpy(Q_t).float().to(x0_pred_prob.device)
    else:
      Q_t = torch.eye(2).float().to(x0_pred_prob.device)
    Q_bar_t_source = torch.from_numpy(diffusion.Q_bar[t]).float().to(x0_pred_prob.device)
    Q_bar_t_target = torch.from_numpy(diffusion.Q_bar[target_t]).float().to(x0_pred_prob.device)

    xt = F.one_hot(xt.long(), num_classes=2).float()
    xt = xt.reshape(x0_pred_prob.shape)

    x_t_target_prob_part_1 = torch.matmul(xt, Q_t.permute((1, 0)).contiguous())
    x_t_target_prob_part_2 = Q_bar_t_target[0]
    x_t_target_prob_part_3 = (Q_bar_t_source[0] * xt).sum(dim=-1, keepdim=True)

    x_t_target_prob = (x_t_target_prob_part_1 * x_t_target_prob_part_2) / x_t_target_prob_part_3

    sum_x_t_target_prob = x_t_target_prob[..., 1] * x0_pred_prob[..., 0]
    x_t_target_prob_part_2_new = Q_bar_t_target[1]
    x_t_target_prob_part_3_new = (Q_bar_t_source[1] * xt).sum(dim=-1, keepdim=True)

    x_t_source_prob_new = (x_t_target_prob_part_1 * x_t_target_prob_part_2_new) / x_t_target_prob_part_3_new

    sum_x_t_target_prob += x_t_source_prob_new[..., 1] * x0_pred_prob[..., 1]

    if target_t > 0:
      xt = torch.bernoulli(sum_x_t_target_prob.clamp(0, 1))
    else:
      xt = sum_x_t_target_prob.clamp(min=0)

    if self.sparse:
      xt = xt.reshape(-1)
    return xt

  def gaussian_posterior(self, target_t, t, pred, xt):
    """Sample (or deterministically denoise) from the Gaussian posterior for a given time step.
       See https://arxiv.org/pdf/2010.02502.pdf for details.
    """
    diffusion = self.diffusion
    if target_t is None:
      target_t = t - 1
    else:
      target_t = torch.from_numpy(target_t).view(1)

    atbar = diffusion.alphabar[t]
    atbar_target = diffusion.alphabar[target_t]

    if self.args.inference_trick is None or t <= 1:
      # Use DDPM posterior
      at = diffusion.alpha[t]
      z = torch.randn_like(xt)
      atbar_prev = diffusion.alphabar[t - 1]
      beta_tilde = diffusion.beta[t - 1] * (1 - atbar_prev) / (1 - atbar)

      xt_target = (1 / np.sqrt(at)).item() * (xt - ((1 - at) / np.sqrt(1 - atbar)).item() * pred)
      xt_target = xt_target + np.sqrt(beta_tilde).item() * z
    elif self.args.inference_trick == 'ddim':
      xt_target = np.sqrt(atbar_target / atbar).item() * (xt - np.sqrt(1 - atbar).item() * pred)
      xt_target = xt_target + np.sqrt(1 - atbar_target).item() * pred
    else:
      raise ValueError('Unknown inference trick {}'.format(self.args.inference_trick))
    return xt_target

  def duplicate_edge_index(self, edge_index, num_nodes, device):
    """Duplicate the edge index (in sparse graphs) for parallel sampling."""
    edge_index = edge_index.reshape((2, 1, -1))
    edge_index_indent = torch.arange(0, self.args.parallel_sampling).view(1, -1, 1).to(device)
    edge_index_indent = edge_index_indent * num_nodes
    edge_index = edge_index + edge_index_indent
    edge_index = edge_index.reshape((2, -1))
    return edge_index
