import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from transformers import get_cosine_schedule_with_warmup
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
import math


import numpy as np
from torch.optim.lr_scheduler import _LRScheduler


class CosineLR(_LRScheduler):
    """SGD with cosine annealing.
    """

    def __init__(self, optimizer, step_size_min=1e-5, t0=100, tmult=2, curr_epoch=-1, last_epoch=-1):
        self.step_size_min = step_size_min
        self.t0 = t0
        self.tmult = tmult
        self.epochs_since_restart = curr_epoch
        super(CosineLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        self.epochs_since_restart += 1

        if self.epochs_since_restart > self.t0:
            self.t0 *= self.tmult
            self.epochs_since_restart = 0

        lrs = [self.step_size_min + (
                    0.5 * (base_lr - self.step_size_min) * (1 + np.cos(self.epochs_since_restart * np.pi / self.t0)))
               for base_lr in self.base_lrs]
        return lrs
    
    
def warmup_linear_decay(step, config):
    warm_up_step = config['lr_scheduler']['WarmUpLinearDecay']['warm_up_step']
    train_steps  = config['lr_scheduler']['WarmUpLinearDecay']['train_steps']
    if step < warm_up_step:
        return (step+1)/warm_up_step
    elif step < train_steps:
        return (train_steps-step)/(train_steps-warm_up_step)
    else:
        return 1.0/(train_steps-warm_up_step)
#scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_linear_decay)



def step_lr(step, config):
    milestones  = config['lr_scheduler']['StepLR']['milestones']
    multipliers = config['lr_scheduler']['StepLR']['multipliers']
    n = len(milestones)
    mul = 1
    for i in range(n):
        if step>=milestones[i]:
            mul = multipliers[i]
    return mul










def fetch_scheduler(optimizer, cfg, df):
    if cfg.training.scheduler == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            # T_max= int(30000/cfg.training.batch_size*cfg.training.epochs)+50,
            T_max=  int(len(df) / cfg.training.batch_size * cfg.training.epochs), 
            eta_min=cfg.training.min_lr
        )
        
    elif cfg.training.scheduler == 'CosineAnnealingWarmRestarts':
        
        num_train_steps = int(len(df) / cfg.training.batch_size * cfg.training.epochs)
        num_warmup_steps = num_train_steps * cfg.training.warmup_ratio
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, 
            num_warmup_steps = cfg.training.warmup_epoch, #num_warmup_steps, 
            num_training_steps = int(len(df) / cfg.training.batch_size * cfg.training.epochs), 
            num_cycles = cfg.training.num_cycles
        )
        # scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=cfg.training.T_0, 
                                                            #  eta_min=cfg.training.min_lr)
    elif cfg.training.scheduler == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=7,
            threshold=0.0001,
            min_lr=cfg.training.min_lr,
        )
    elif cfg.training.scheduler == 'ExponentialLR':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
    elif cfg.training.scheduler == None:
        return None
    return scheduler



def cosine_warmup_lr_scheduler(optimizer, warmup_iters, max_iters, warmup_factor):
    def lr_lambda(current_iter):
        if current_iter < warmup_iters:
            return float(current_iter) / float(max(1.0, warmup_iters)) * warmup_factor
        else:
            progress = float(current_iter - warmup_iters) / float(max(1, max_iters - warmup_iters))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)