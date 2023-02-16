"""Misc. optimizer implementations."""
from functools import partial

import torch
from torch.optim.lr_scheduler import LambdaLR


def get_schedule_fn(scheduler, num_training_steps):
    """Returns a callable scheduler_fn(optimizer).
    Todo: Sanitize and unify these schedulers...
    """
    if scheduler == "cosine-decay":
        scheduler_fn = partial(
            torch.optim.lr_scheduler.CosineAnnealingLR,
            T_max=num_training_steps,
            eta_min=0.0,
        )
    elif scheduler == "one-cycle":  # this is a simplified one-cycle
        scheduler_fn = partial(
            get_one_cycle,
            num_training_steps=num_training_steps,
        )
    else:
        raise ValueError(f"Invalid schedule {scheduler} given.")
    return scheduler_fn


def get_one_cycle(optimizer, num_training_steps):
    """Simple single-cycle scheduler. Not including paper/fastai three-phase things or asymmetry."""

    def lr_lambda(current_step):
        if current_step < num_training_steps / 2:
            return float(current_step / (num_training_steps / 2))
        else:
            return float(2 - current_step / (num_training_steps / 2))

    return LambdaLR(optimizer, lr_lambda, -1)
