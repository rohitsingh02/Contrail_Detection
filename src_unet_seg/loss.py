import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
from torch.autograd import Variable



JaccardLoss = smp.losses.JaccardLoss(mode='binary')
DiceLoss    = smp.losses.DiceLoss(mode='binary')
BCELoss     = smp.losses.SoftBCEWithLogitsLoss()
BCELossPos  = smp.losses.SoftBCEWithLogitsLoss(ignore_index=0, smooth_factor=0.1)
# ignore_index – Specifies a target value that is ignored and does not contribute to the input gradient.
# smooth_factor – Factor to smooth target (e.g. if smooth_factor=0.1 then [1, 0, 1] -> [0.9, 0.1, 0.9])

LovaszLoss  = smp.losses.LovaszLoss(mode='binary', per_image=False)
TverskyLoss = smp.losses.TverskyLoss(mode='binary', log_loss=False)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, logit, target):
        target = target.float()
        max_val = (-logit).clamp(min=0)
        loss = logit - logit * target + max_val + \
               ((-max_val).exp() + (-logit - max_val).exp()).log()

        invprobs = F.logsigmoid(-logit * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        if len(loss.size())==2:
            loss = loss.sum(dim=1)
        return loss.mean()


def dice_coef(y_true, y_pred, thr=0.5, epsilon=1e-6):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred>thr).to(torch.float32)
    inter = (y_true*y_pred).sum()
    den = y_true.sum() + y_pred.sum()
    dice = ((2*inter+epsilon)/(den+epsilon)).mean()
    return dice


def iou_coef(y_true, y_pred, thr=0.5, epsilon=1e-6):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred>thr).to(torch.float32)
    inter = (y_true*y_pred).sum()
    union = (y_true + y_pred - y_true*y_pred).sum()
    iou = ((inter+epsilon)/(union+epsilon)).mean(0)
    return iou


def dice_loss(input, target):
    input = torch.sigmoid(input)
    smooth = 1.0
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    return ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))


class MixedLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss(gamma)

    def forward(self, input, target):
        loss = self.alpha*self.focal(input, target) - torch.log(dice_loss(input, target))
        return loss.mean()



def loss_fn(y_pred, y_true):
    return 0.5*BCELoss(y_pred, y_true) + 0.5*TverskyLoss(y_pred, y_true)


def loss_fn_s2(y_pred, y_true):
    mixed_loss = MixedLoss()
    return mixed_loss(y_pred, y_true) #0.5*DiceLoss(y_pred, y_true) + 0.5*TverskyLoss(y_pred, y_true)   #DiceLoss(y_pred, y_true) #+ 0.5*LovaszLoss(y_pred, y_true) # soft_dice_criterion(y_pred, y_true)    #BCELoss(y_pred, y_true) #JaccardLoss(y_pred, y_true)
    

def loss_fn_s3(y_pred, y_true):
    return DiceLoss(y_pred, y_true)