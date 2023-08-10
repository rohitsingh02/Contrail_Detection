import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import cv2

    
      
def get_transform(cfg):
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
            A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.5),
            A.GridDistortion(num_steps=2, distort_limit=0.05, p=0.2),
            A.CoarseDropout(max_holes=8, max_height=cfg.dataset.img_height//20, max_width=cfg.dataset.img_width//20,
                            min_holes=5, fill_value=0, 
                            # mask_fill_value=0,
                            p=0.5),
            ], p=1.0),
        
        "train_3": A.Compose([
            A.Resize( height=cfg.dataset.img_height + 128, width=cfg.dataset.img_width + 128, interpolation=cv2.INTER_NEAREST),            
            A.RandomResizedCrop(height=cfg.dataset.img_height, width=cfg.dataset.img_width),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.5),
            A.GridDistortion(num_steps=2, distort_limit=0.05, p=0.2),
            A.CoarseDropout(max_holes=8, max_height=cfg.dataset.img_height//20, max_width=cfg.dataset.img_width//20,
                            min_holes=5, fill_value=0, 
                            # mask_fill_value=0,
                            p=0.5),
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