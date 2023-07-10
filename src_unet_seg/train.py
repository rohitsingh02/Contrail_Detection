import os
import shutil
import pathlib
import time
import re
import gc
import shutil
import yaml
import sys
import argparse
from PIL import Image
import pandas as pd
import numpy as np
import cv2 as cv
import random
import importlib
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset
import torchvision
from types import SimpleNamespace

from torchvision import datasets
import cv2
from torch.cuda import amp

import torchvision.transforms as T
from torchvision.transforms import Compose, ToTensor, Resize
from torchvision.utils import make_grid
from schedulers import fetch_scheduler, cosine_warmup_lr_scheduler
from loss import loss_fn, loss_fn_s2, loss_fn_s3, dice_coef, iou_coef
import segmentation_models_pytorch as smp
from config import get_config
from schedulers import CosineLR
from losses.loss import dice_loss_non_empty_masks, dice_loss,  criterion_lovasz_hinge_non_empty
from losses.lovasz_loss import lovasz_hinge 

import augs
import utils
import warnings

sys.path.append("models")
sys.path.append("datasets")


def mixup(input, truth, clip=[0, 1]):
    indices = torch.randperm(input.size(0))
    shuffled_input = input[indices]
    shuffled_labels = truth[indices]

    lam = np.random.uniform(clip[0], clip[1])
    input = input * lam + shuffled_input * (1 - lam)
    return input, truth, shuffled_labels, lam

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int_(W * cut_rat)
    cut_h = np.int_(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def cutmix(data, target, alpha=1.):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_target = target[indices]

    lam = np.clip(np.random.beta(alpha, alpha),0.3,0.4)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    new_data = data.clone()
    new_data[:, :, bby1:bby2, bbx1:bbx2] = data[indices, :, bby1:bby2, bbx1:bbx2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
    targets = (target, shuffled_target, lam)

    return new_data,target,shuffled_target,lam


# =========================================================================================
# get model
# ========================================================================================= 
def get_model(cfg):
    # if cfg.architecture.backbone == "se_resnext101_32x4d":
    #     Net = importlib.import_module(cfg.model_class).UNET_SERESNEXT101
    if hasattr(cfg, "model_type") and cfg.model_type == "timm_unet":
        Net = importlib.import_module(cfg.model_class).TimmSegModel
    else:
        Net = importlib.import_module(cfg.model_class).Net
        
    return Net(cfg)


def load_checkpoint(cfg, model):
    weight = f"{cfg.architecture.pretrained_weights}"
    
    if cfg.dataset.fold != -1:
       weight += f"/checkpoint_dice_ctrl_fold{cfg.dataset.fold}.pth" 
    
    d =  torch.load(weight, map_location="cpu")
    if "model" in d:
        model_weights = d["model"]
    else:
        model_weights = d
    try:
        model.load_state_dict(model_weights, strict=True)
    except Exception as e:
        print("removing unused pretrained layers")
        for layer_name in re.findall("size mismatch for (.*?):", str(e)):
            model_weights.pop(layer_name, None)
        model.load_state_dict(model_weights, strict=False)

    print(f"Weights loaded from: {cfg.architecture.pretrained_weights}")
        
# =========================================================================================
# Train Func
# =========================================================================================
        
def train_one_epoch(cfg, config, model, optimizer, scheduler, criterion, dataloader):
    
    losses = utils.AverageMeter()
    model.train()
    scaler = amp.GradScaler()
    
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Train ')
    for step, (images, masks, labels_cls) in pbar:         
        images = images.to(cfg.device, dtype=torch.float)
        masks  = masks.to(cfg.device, dtype=torch.float)
        labels_cls = labels_cls.to(device, torch.float)
        
        mixing = random.random()
        do_mixup, do_cutmix = False, False
        if hasattr(cfg.dataset, "mixup") and cfg.dataset.mixup:
            if mixing < 0.25:
                do_mixup = True
                images, masks, masks_sfl, lam = mixup(images, masks)

        if hasattr(cfg.dataset, "cutmix") and cfg.dataset.cutmix:
            if (mixing>0.25) and (mixing <0.5) :
                do_cutmix = True
                images, masks, masks_sfl, lam = cutmix(images, masks)  

        
        
        
        batch_size,c,h,w = images.shape
        
        with amp.autocast(enabled=True):
            
            # if cfg.architecture.backbone == "se_resnext101_32x4d":
            #     # y_pred, y_pred_deeps, y_pred_cls =  model(images)
            #     y_pred =  model(images)
                
            if hasattr(cfg, "model_type") and cfg.model_type == "timm_unet":
                y_pred = model(images) 
                # y_pred, y_pred_cls, y_pred_pix = model(images) 
            else:
                y_pred = model(images)    
            
            
            loss  = criterion(y_pred, masks)
            if do_mixup or do_cutmix:
                loss2  = criterion(y_pred, masks_sfl)
                loss = loss * lam  + loss2 * (1 - lam)                 
            

            if hasattr(cfg.training, 'loss') and cfg.training.loss == "bce":
                loss  = criterion(y_pred, masks)
            elif hasattr(cfg.training, 'loss') and cfg.training.loss == "bce+mixed_loss":
                loss  = criterion(y_pred, masks)
                loss += loss_fn_s2(y_pred, masks) 
            elif hasattr(cfg.training, 'loss') and cfg.training.loss == "bce+lovastz":
                loss  = criterion(y_pred, masks)
                loss += lovasz_hinge(y_pred.view(-1,h,w), masks.view(-1,h,w))
            elif hasattr(cfg.training, 'loss') and cfg.training.loss == "bce+tversky":
                loss += loss_fn(y_pred, masks) 
            elif  hasattr(cfg.training, 'loss') and cfg.training.loss == "dice": 
                loss += loss_fn_s3(y_pred, masks)
              
            
        if cfg.training.grad_accumulation > 1:
            loss = loss / cfg.training.grad_accumulation
        losses.update(loss.item(), batch_size)
                      
        scaler.scale(loss).backward()
        if (step + 1) % cfg.training.grad_accumulation == 0:
            scaler.step(optimizer)
            scaler.update()
            # zero the parameter gradients
            optimizer.zero_grad()
            if config['lr_scheduler_name']=='OneCycleLR':
                scheduler.step()
        
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(
            fold = cfg.dataset.fold,
            epoch=cfg.epoch,
            train_loss=f'{losses.avg:0.4f}',
            lr=f'{current_lr:0.5f}',
            gpu_mem=f'{mem:0.2f} GB'
        )
        
        
    cfg.logger.info(
        f"Fold {cfg.dataset.fold} Epoch: {cfg.epoch}, train_loss: {losses.avg:0.4f}, lr: {current_lr:0.5f}, gpu_mem: {mem:0.2f} GB"
    ) 
        
    torch.cuda.empty_cache()
    gc.collect()
    return losses.avg


@torch.no_grad()
def valid_one_epoch(cfg, model, optimizer, criterion, dataloader):
    losses = utils.AverageMeter()
    
    model.eval()
    val_preds = []
    val_scores = []
    outputs = []
    masks_ = []
    
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Valid ')
    for step, (images, masks, labels_cls) in pbar:        
        images  = images.to(cfg.device, dtype=torch.float)
        masks   = masks.to(cfg.device, dtype=torch.float)
        labels_cls = labels_cls.to(device, torch.float)
        masks_.append(torch.squeeze(masks, dim=1))
                
        batch_size, c, h, w = images.shape        
        # if cfg.architecture.backbone == "se_resnext101_32x4d":
        #     # y_pred, y_pred_deeps, y_pred_cls =  model(images)
        #     # print(type(images))
        #     y_pred =  model(images)
            
            
        if hasattr(cfg, "model_type") and cfg.model_type == "timm_unet":
            y_pred = model(images) 
            # y_pred, y_pred_cls, y_pred_pix = model(images) 
        else:
            y_pred = model(images)
            
        loss  = criterion(y_pred, masks)
        
        if hasattr(cfg.training, 'loss') and cfg.training.loss == "bce":
            loss  = criterion(y_pred, masks)
        elif hasattr(cfg.training, 'loss') and cfg.training.loss == "bce+mixed_loss": # all 50 epochs so far
            loss  = criterion(y_pred, masks)
            loss += loss_fn_s2(y_pred, masks) 
        elif hasattr(cfg.training, 'loss') and cfg.training.loss == "bce+lovastz":
            loss  = criterion(y_pred, masks)
            loss += lovasz_hinge(y_pred.view(-1,h,w), masks.view(-1,h,w))
        elif hasattr(cfg.training, 'loss') and cfg.training.loss == "bce+tversky":
            loss += loss_fn(y_pred, masks) 
        elif  hasattr(cfg.training, 'loss') and cfg.training.loss == "dice":
            loss += loss_fn_s3(y_pred, masks)
        
                    
        losses.update(loss.item(), batch_size)
        # if cfg.dataset.img_height > 512:
        #     y_pred = torch.nn.functional.interpolate(y_pred, size=256, mode='nearest') 

        
        y_pred = y_pred.sigmoid()
        val_preds.append(torch.squeeze(y_pred, dim=1))
                
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(
            fold=cfg.dataset.fold,
            epoch=cfg.epoch,
            valid_loss=f'{losses.avg:0.4f}',            
            # val_score=np.mean(val_scores, axis=0),
            lr=f'{current_lr:0.5f}',
            gpu_memory=f'{mem:0.2f} GB'
        )
        
    model_masks = torch.cat(masks_, dim=0)
    model_preds = torch.cat(val_preds, dim=0)
    model_masks = torch.flatten(model_masks, start_dim=0, end_dim=1)
    model_preds = torch.flatten(model_preds, start_dim=0, end_dim=1) 
    
    
    # best_threshold = 0.0
    # best_dice_score = 0.0
    # for threshold in [i / 100 for i in range(101)] :
    #     # if threshold < 0.1 or threshold > 0.55:
    #     #     continue
    #     score = dice_coef(model_masks, model_preds, thr=threshold).cpu().detach().numpy() 
    #     if score > best_dice_score:
    #         best_dice_score = score
    #         best_threshold = threshold
    
    
    best_dice_score = dice_coef(model_masks, model_preds).cpu().detach().numpy() 
    # dice_score = dice_coef(model_masks, model_preds).cpu().detach().numpy() 
    iou_score = iou_coef(model_masks, model_preds).cpu().detach().numpy()        
    val_scores = [best_dice_score, iou_score]
        
    
    cfg.logger.info(
        f"Epoch: {cfg.epoch}, val_loss: {losses.avg:0.4f}, dice_score: {best_dice_score.item():0.4f}, dice_th: 0.5, iou_score: {iou_score.item():0.4f}, lr: {current_lr:0.5f}, gpu_mem: {mem:0.2f} GB"
    )
        
    torch.cuda.empty_cache()
    gc.collect()
    # return losses.avg, val_preds, val_scores
    return losses.avg, val_scores


# =========================================================================================
# Train & Evaluate
# =========================================================================================
def train_loop(train_df, val_df, val_df_contrail, cfg, config):
    print(' ')
    print(f"========== Training ==========")
    print(train_df.shape, val_df.shape, val_df_contrail.shape)    
    
    if cfg.debug: 
        print("DEBUG MODE")
        train_df = train_df.head(50)
    
    
    criterion = nn.BCEWithLogitsLoss().to(device)
    
    if hasattr(cfg.training, 'augs_train') and cfg.training.augs_train == "train_1":
        train_transform =  augs.get_transform(cfg=cfg)['train_1']
    elif hasattr(cfg.training, 'augs_train') and cfg.training.augs_train == "train_2":
        train_transform =  augs.get_transform(cfg=cfg)['train_2']
    else:
        train_transform =  augs.get_transform(cfg=cfg)['train_1']

    if hasattr(cfg.training, 'augs_val') and cfg.training.augs_val == "valid_1":
        val_transform =  augs.get_transform(cfg=cfg)['valid_1']
    elif hasattr(cfg.training, 'augs_val') and cfg.training.augs_val == "valid_2":
        val_transform =  augs.get_transform(cfg=cfg)['valid_2']
    else:
        val_transform =  augs.get_transform(cfg=cfg)['valid_2']
    
                    
    train_dataset = cfg.CustomDataset(train_df, cfg, transform=train_transform, mode="train")
    # valid_dataset = cfg.CustomDataset(val_df, cfg, transform=val_transform, mode="val")  
    valid_dataset_ctrl = cfg.CustomDataset(val_df_contrail, cfg, transform=val_transform, mode="val")      

    train_loader = DataLoader(
        train_dataset, 
        batch_size = cfg.training.batch_size, 
        shuffle = True, 
        num_workers = cfg.environment.num_workers, 
        pin_memory = True, 
        drop_last = True,
    )

    # valid_loader = DataLoader(
    #     valid_dataset, 
    #     batch_size = cfg.training.batch_size * 2, 
    #     shuffle = False, 
    #     num_workers = cfg.environment.num_workers, 
    #     pin_memory = True, 
    #     drop_last = False
    # )
    
    valid_loader_ctrl = DataLoader(
        valid_dataset_ctrl, 
        batch_size = cfg.training.batch_size * 2, 
        shuffle = False, 
        num_workers = cfg.environment.num_workers, 
        pin_memory = True, 
        drop_last = False
    )
    
    
    # init and load model
    model = get_model(cfg)   
    model.to(cfg.device)
    model = torch.nn.DataParallel(model)
    if cfg.architecture.pretrained_weights != "":
        load_checkpoint(cfg, model)
    model.to(cfg.device)
    
    # init optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), **config['Adam'])
    if config['lr_scheduler_name']=='ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **config['lr_scheduler']['ReduceLROnPlateau'])
    elif config['lr_scheduler_name']=='CosineAnnealingLR':
        scheduler = CosineLR(optimizer, **config['lr_scheduler']['CosineAnnealingLR'])
        print(scheduler.get_lr())
    elif config['lr_scheduler_name']=='OneCycleLR':
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, steps_per_epoch=len(train_loader),
                                                    **config['lr_scheduler']['OneCycleLR'])
    
    # Training & Validation loop
    best_dice = -np.inf
    best_dice_ctrl = -np.inf
        
    for epoch in range(cfg.training.epochs):
        
        print(cfg.dataset.fold)
        
        cfg.epoch = epoch                
        train_loss = train_one_epoch(cfg, config, model, optimizer, scheduler, criterion, train_loader)         
        
        # val_loss, val_scores = valid_one_epoch(cfg, model, optimizer, criterion, valid_loader)    
        val_loss_ctrl, val_scores_ctrl = valid_one_epoch(cfg, model, optimizer, criterion, valid_loader_ctrl)    

        # val_dice, val_dice_ctrl = val_scores[0], val_scores_ctrl[0]
        val_dice_ctrl = val_scores_ctrl[0]
        
        if config['lr_scheduler_name']=='ReduceLROnPlateau':
            scheduler.step(val_loss_ctrl)
        elif config['lr_scheduler_name']=='CosineAnnealingLR':
            scheduler.step()
        
        
        # # deep copy the model
        # if val_dice >= best_dice:
        #     cfg.logger.info(f"Fold {cfg.dataset.fold} - Epoch {epoch} - Valid Dice Score Improved ({best_dice:0.4f} ---> {val_dice:0.4f})")
        #     best_dice = val_dice
        #     model_save_pth = ""
        #     if cfg.dataset.fold != -1:
        #         model_save_pth = f"{cfg.output_dir}/checkpoint_dice_fold{cfg.dataset.fold}.pth"
        #     else:
        #         model_save_pth = f"{cfg.output_dir}/checkpoint_dice.pth"  
        #     ## calculate and save statistics
        #     torch.save( {'model': model.state_dict()}, model_save_pth)   
            
            
        # deep copy the model
        if val_dice_ctrl >= best_dice_ctrl:
            cfg.logger.info(f"Fold {cfg.dataset.fold} - Epoch {epoch} - Valid Dice CTRL Score Improved ({best_dice_ctrl:0.4f} ---> {val_dice_ctrl:0.4f})")
            best_dice_ctrl = val_dice_ctrl
            model_save_pth = ""
            if cfg.dataset.fold != -1:
                model_save_pth = f"{cfg.output_dir}/checkpoint_dice_ctrl_fold{cfg.dataset.fold}.pth"
            else:
                model_save_pth = f"{cfg.output_dir}/checkpoint_dice_ctrl.pth"  
            ## calculate and save statistics
            torch.save( {'model': model.state_dict()}, model_save_pth)   
            
        
    cfg.logger.info(f"Fold {cfg.dataset.fold} - Dice CV ({best_dice_ctrl:0.4f})")
    cfg.logger.info("*"*100)

    print("*"*100)
    torch.cuda.empty_cache()
    gc.collect()
    return val_df



# setting up config
parser = argparse.ArgumentParser(description="")
parser.add_argument("-C", "--config", help="config filename")

parser_args, _ = parser.parse_known_args(sys.argv)
cfg = yaml.safe_load(open(parser_args.config).read())
for k, v in cfg.items():
    if type(v) == dict:
        cfg[k] = SimpleNamespace(**v)
cfg = SimpleNamespace(**cfg)


if hasattr(cfg, 'stage'):
    cfg.training.epochs = 50  #15 #5
    output_dir = f"{cfg.exp_name}/{cfg.architecture.model_name}/{cfg.architecture.backbone.replace('/', '-')}/{cfg.stage}"
else:
    if hasattr(cfg.training, "stage"):
        output_dir = f"{cfg.exp_name}/{cfg.architecture.model_name}/{cfg.architecture.backbone.replace('/', '-')}-{cfg.dataset.img_height}-{cfg.training.stage}"
    else:
        output_dir = f"{cfg.exp_name}/{cfg.architecture.model_name}/{cfg.architecture.backbone.replace('/', '-')}-{cfg.dataset.img_height}"


cfg.output_dir = output_dir
model_save_path = os.makedirs(output_dir, exist_ok=True)
shutil.copy(parser_args.config, cfg.output_dir)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cfg.device = device
cfg.CustomDataset = importlib.import_module(cfg.dataset_class).ContrailDataset


if __name__ == "__main__":
    print("Start Training....")
    utils.seed_everything(cfg.environment.seed)
    cfg.logger = utils.get_logger(cfg=cfg)
    
    cfg.logger.info("*"*100)
    config = get_config()
    
    config['lr_scheduler']['CosineAnnealingLR']['t0'] = cfg.training.epochs - 1
    if hasattr(cfg.training, 'lr'):
        config['Adam']['lr'] = cfg.training.lr

    df = pd.read_csv('../input/data_utils/train_5_folds.csv')
    val_df_contrail = pd.read_csv('../input/data_utils/val_df_filled.csv') 
    train_dups = np.load("/home/rohits/pv1/Contrail_Detection/input/data_utils/dups_train.npy")
    train_dups = [int(train_id) for train_id in train_dups]
    
    if hasattr(cfg.dataset, "remove_outleirs") and cfg.dataset.remove_outleirs:
        df['id'] = df['id'].apply(lambda x: str(x))
        val_df_contrail['id'] = val_df_contrail['id'].apply(lambda x: str(x))
        print(f"Shape Before Removing Outliers: {df.shape}, {val_df_contrail.shape}" )
        train_outs = np.load('../input/data_utils/train_outliers.npy')
        val_outs = np.load('../input/data_utils/val_outliers.npy')
        df = df.loc[~df['id'].isin(list(train_outs))].reset_index(drop=True)
        val_df_contrail = val_df_contrail.loc[~val_df_contrail['id'].isin(list(val_outs))].reset_index(drop=True)
        print(f"Shape After Removing Outliers: {df.shape}, {val_df_contrail.shape}" )

        
    for fold in [0]: # [0, 1, 2, 3, 4]
        cfg.dataset.fold = fold
        train_df = df.loc[df.fold != fold].reset_index(drop=True)
        val_df = df.loc[df.fold == fold].reset_index(drop=True)
        print(df.shape, train_df.shape, val_df.shape)
        
        
        if hasattr(cfg.training, 'use_pl') and cfg.training.use_pl:
            
            if hasattr(cfg.training, "pl_version") and cfg.training.pl_version == "v1":
                df_tr_s1_3 = pd.read_csv(f'../input/pseudo/train_data_3.csv') # s1
                df_tr_s1_5 = pd.read_csv(f'../input/pseudo/train_data_5.csv') # s1

                df_tr_s1_3 = df_tr_s1_3.loc[df_tr_s1_3.fold != fold].reset_index(drop=True)
                df_tr_s1_5 = df_tr_s1_5.loc[df_tr_s1_5.fold != fold].reset_index(drop=True)

                df_tr_s1_3['label'] = df_tr_s1_3['label'].apply(lambda x: f"{x.split('.npy')[0]}_5fold_s1.npy")
                df_tr_s1_5['label'] = df_tr_s1_5['label'].apply(lambda x: f"{x.split('.npy')[0]}_5fold_s1.npy")

                
                df_val_s1_3 = pd.read_csv(f'../input/pseudo/validation_data_3.csv') # s1
                df_val_s1_5 = pd.read_csv(f'../input/pseudo/validation_data_5.csv') # s1

                df_val_s1_3['label'] = df_val_s1_3['label'].apply(lambda x: f"{x.split('.npy')[0]}_5fold_s1.npy")
                df_val_s1_5['label'] = df_val_s1_5['label'].apply(lambda x: f"{x.split('.npy')[0]}_5fold_s1.npy")

                train_df = pd.concat([
                    train_df, df_tr_s1_3, df_val_s1_5, df_val_s1_3, df_val_s1_5
                ]).reset_index(drop=True) 
            elif hasattr(cfg.training, "pl_version") and cfg.training.pl_version == "v2":
                df_val_s1_2 = pd.read_csv(f'../input/pseudo/validation_data_2.csv') # s1
                df_val_s1_3 = pd.read_csv(f'../input/pseudo/validation_data_3.csv') # s1
                df_val_s1_5 = pd.read_csv(f'../input/pseudo/validation_data_5.csv') # s1
                df_val_s1_6 = pd.read_csv(f'../input/pseudo/validation_data_6.csv') # s1
                
                df_val_s1_2['label'] = df_val_s1_2['label'].apply(lambda x: f"{x.split('.npy')[0]}_684cv_687lb.npy")
                df_val_s1_3['label'] = df_val_s1_3['label'].apply(lambda x: f"{x.split('.npy')[0]}_684cv_687lb.npy")
                df_val_s1_5['label'] = df_val_s1_5['label'].apply(lambda x: f"{x.split('.npy')[0]}_684cv_687lb.npy")
                df_val_s1_6['label'] = df_val_s1_6['label'].apply(lambda x: f"{x.split('.npy')[0]}_684cv_687lb.npy")
                
                
                train_df = pd.concat([
                    train_df, df_val_s1_2, df_val_s1_3, df_val_s1_5, df_val_s1_6
                ]).reset_index(drop=True) 
                
            elif hasattr(cfg.training, "pl_version") and cfg.training.pl_version == "v3":
                
                df_tr_s1_3 = pd.read_csv(f'../input/pseudo/train_data_3.csv') # s1
                df_tr_s1_5 = pd.read_csv(f'../input/pseudo/train_data_5.csv') # s1
                
                
                df_val_s1_2 = pd.read_csv(f'../input/pseudo/validation_data_2.csv') # s1
                df_val_s1_3 = pd.read_csv(f'../input/pseudo/validation_data_3.csv') # s1
                df_val_s1_5 = pd.read_csv(f'../input/pseudo/validation_data_5.csv') # s1
                df_val_s1_6 = pd.read_csv(f'../input/pseudo/validation_data_6.csv') # s1
                
                df_tr_s1_3['label'] = df_tr_s1_3['label'].apply(lambda x: f"{x.split('.npy')[0]}_684cv_687lb.npy")
                df_tr_s1_5['label'] = df_tr_s1_5['label'].apply(lambda x: f"{x.split('.npy')[0]}_684cv_687lb.npy")
                
                
                df_val_s1_2['label'] = df_val_s1_2['label'].apply(lambda x: f"{x.split('.npy')[0]}_684cv_687lb.npy")
                df_val_s1_3['label'] = df_val_s1_3['label'].apply(lambda x: f"{x.split('.npy')[0]}_684cv_687lb.npy")
                df_val_s1_5['label'] = df_val_s1_5['label'].apply(lambda x: f"{x.split('.npy')[0]}_684cv_687lb.npy")
                df_val_s1_6['label'] = df_val_s1_6['label'].apply(lambda x: f"{x.split('.npy')[0]}_684cv_687lb.npy")
                
                
                # train_df = pd.concat([
                #     train_df, df_tr_s1_3, df_tr_s1_5, df_val_s1_2, df_val_s1_3, df_val_s1_5, df_val_s1_6
                # ]).reset_index(drop=True) 
                
                train_df = pd.concat([
                    train_df, df_tr_s1_3, df_val_s1_3, df_val_s1_5
                ]).reset_index(drop=True) 
                
                
                
            elif hasattr(cfg.training, "pl_version") and cfg.training.pl_version == "v4":
                df_tr_s1_3 = pd.read_csv(f'../input/pseudo/train_data_3.csv') # s1
                df_tr_s1_3['label'] = df_tr_s1_3['label'].apply(lambda x: f"{x.split('.npy')[0]}_6811_702lb.npy")
                train_df = pd.concat([
                    train_df, df_tr_s1_3
                ]).reset_index(drop=True) 
 
 
        print(train_df.shape)
        print(train_df['class'].value_counts())
        
        train_df = pd.concat([train_df, val_df]).reset_index(drop=True)
        # print(train_df.shape)
        train_df = train_df.loc[~train_df['id'].isin(train_dups)].reset_index(drop=True)
        # print(train_df.shape)
        
        
        val_df = val_df_contrail
        _ = train_loop(train_df, val_df, val_df_contrail, cfg, config)
    
    
