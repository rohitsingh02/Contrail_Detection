import random
import os
import torch

VERSION = '02'

def get_config():
    config = { 
              
        'model_bs': {
            'efficientnet-b7-256': 96,
            'efficientnet-b7-384': 42,
            'efficientnet-b7-512': 24,
            'timm-efficientnet-b8-256': 68,
            'timm-efficientnet-l2-256': 18
              
        },
              
                           
        'Adam':{
            'lr': 2e-3,  #1e-4, ps_01
            'betas':(0.9, 0.999),
            'weight_decay':1e-5,
        },
        'SGD':{
            'lr':0.01,
            'momentum':0.9,
        },
        'lr_scheduler_name':'CosineAnnealingLR', #'OneCycleLR', #'ReduceLROnPlateau', #'StepLR',#'WarmUpLinearDecay', 
        'lr_scheduler':{
            'ReduceLROnPlateau':{
                'factor':0.8,
                'patience':5,
                'min_lr':1e-5,
                'verbose':True,
            },
            'OneCycleLR':{
                'pct_start':0.1,
                'div_factor':1e3, 
                'max_lr':1e-2,
                'epochs':25,
            },
            'CosineAnnealingLR':{
                'step_size_min':1e-4,
                't0': 49, #49,
                'tmult':1,
                'curr_epoch':-1,
                'last_epoch':-1,
            },
            'WarmUpLinearDecay':{
                'train_steps':40,
                'warm_up_step':3,
            },
            'StepLR':{
                'milestones':[1,2,3,20,40],
                'multipliers':[0.5,0.3,0.1,0.03,0.003],
            },
        },
    }
    return config