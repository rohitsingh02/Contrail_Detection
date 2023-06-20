import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import cv2

    
    
    
def get_transform(cfg):
    
    if cfg.dataset.img_height > 512:
        img_height = 256
        img_width = 256
    else:
        img_height = cfg.dataset.img_height
        img_width = cfg.dataset.img_width
    
    data_transforms = {
        "train_1": A.Compose([
            A.Resize( height=cfg.dataset.img_height, width=cfg.dataset.img_width, interpolation=cv2.INTER_NEAREST),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.5),
            A.GridDistortion(num_steps=2, distort_limit=0.05, p=0.2),
            A.CoarseDropout(max_holes=8, max_height=cfg.dataset.img_height//20, max_width=cfg.dataset.img_width//20,
                            min_holes=5, fill_value=0, 
                            # mask_fill_value=0,
                            p=0.5),
            ], p=1.0),
        
        "train_2": A.Compose([
            A.Resize( height=cfg.dataset.img_height, width=cfg.dataset.img_width, interpolation=cv2.INTER_NEAREST),
            ], p=1.0),
        
        
        "train_3": A.Compose([
            A.Resize( height=cfg.dataset.img_height, width=cfg.dataset.img_width, interpolation=cv2.INTER_NEAREST),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.CoarseDropout(max_holes=4, max_height=cfg.dataset.img_height//4, max_width=cfg.dataset.img_width//4,
                            min_holes=2, fill_value=0, 
                            # mask_fill_value=0,
                            p=0.5),
            ], p=1.0),
        
        
        
        "valid_1": A.Compose([
            A.HorizontalFlip(p=0.5), # from s2 exp
            A.VerticalFlip(p=0.5),
            A.Resize(height=cfg.dataset.img_height, width=cfg.dataset.img_width, interpolation=cv2.INTER_NEAREST),
        ], p=1.0),
        
        "valid_2": A.Compose([
                A.Resize(height=cfg.dataset.img_height, width=cfg.dataset.img_width, interpolation=cv2.INTER_NEAREST),
                # A.Resize(height=img_height, width=img_width, interpolation=cv2.INTER_NEAREST)
        ], p=1.0),
        
    }
    return data_transforms
    
    
    
    
 
import numpy as np
from albumentations import (Compose, HorizontalFlip, VerticalFlip, Rotate, RandomRotate90,
                            ShiftScaleRotate, ElasticTransform,
                            GridDistortion, RandomSizedCrop, RandomCrop, CenterCrop,
                            RandomBrightnessContrast, HueSaturationValue, IAASharpen,
                            RandomGamma, RandomBrightness, RandomBrightnessContrast,
                            GaussianBlur,CLAHE,
                            Cutout, CoarseDropout, GaussNoise, ChannelShuffle, ToGray, OpticalDistortion,
                            Normalize, OneOf, NoOp)

from albumentations.pytorch import ToTensor

MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])

def get_transforms_train(cfg):
    transform_train = Compose([
        A.Resize( height=cfg.dataset.img_height, width=cfg.dataset.img_width, interpolation=cv2.INTER_NEAREST),

        #Basic
        RandomRotate90(p=1),
        HorizontalFlip(p=0.5),
        #Morphology
        ShiftScaleRotate(shift_limit=0, scale_limit=(-0.2,0.2), rotate_limit=(-30,30), 
                         interpolation=1, border_mode=0, value=(0,0,0), p=0.5),
        
        GaussNoise(var_limit=(0,50.0), mean=0, p=0.5),
        GaussianBlur(blur_limit=(3,7), p=0.5),
        
        CoarseDropout(max_holes=2, 
                      max_height=cfg.dataset.img_height//4, max_width=cfg.dataset.img_width//4, 
                      min_holes=1,
                      min_height=cfg.dataset.img_height//16, min_width=cfg.dataset.img_width//16, 
                      fill_value=0, p=0.5),
        
        # Normalize(mean=(MEAN[0], MEAN[1], MEAN[2]), 
        #           std=(STD[0], STD[1], STD[2])),
        # ToTensor(),
    ])
    return transform_train


def get_transforms_valid(cfg):
    transform_valid = Compose([
        A.Resize( height=cfg.dataset.img_height, width=cfg.dataset.img_width, interpolation=cv2.INTER_NEAREST),

        # Normalize(mean=(MEAN[0], MEAN[1], MEAN[2]), 
        #           std=(STD[0], STD[1], STD[2])),
        # ToTensor(),
    ] )
    return transform_valid


def denormalize(z, mean=MEAN.reshape(-1,1,1), std=STD.reshape(-1,1,1)):
    return std*z + mean
