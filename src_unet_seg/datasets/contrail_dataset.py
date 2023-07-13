import torch
import pandas as pd
import numpy as np
import random
import torchvision.transforms as T



# class ContrailDataset:
#     def __init__(self, df, cfg, transform=None, mode="train"):
#         self.df = df  
#         self.cfg = cfg           
#         self.images = df['image']
#         self.labels = df['label']
#         self.transform =transform
#         self.mode=mode        

        
#     def __len__(self):
#         return len(self.images)
        
#     def __getitem__(self, idx):
#         image = np.load("../input/" + self.images[idx]).astype(float)   
#         label = np.load("../input/" + self.labels[idx]).astype(float)
            
#         # label_cls = 1 if label.sum() > 0 else 0
#         if self.transform :
#             data = self.transform(image=image, mask=label)
#             image  = data['image']
#             aug_label  = data['mask']
            
#             image = np.transpose(image, (2, 0, 1))
#             label = np.transpose(aug_label, (2, 0, 1))  
   
#         cls_label = (torch.tensor(aug_label)>0).float()
#         return torch.tensor(image), torch.tensor(label), cls_label

    
    
## This is used after exps 686LB | 68679 CV [19 model weights]
class ContrailDataset:
    def __init__(self, df, cfg, transform=None, mode="train"):
        self.df = df  
        self.cfg = cfg           
        self.images = df['image']
        self.labels = df['label']
        self.transform =transform
        self.mode=mode        
        self.normalize_image = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        image = np.load("../input/" + self.images[idx]).astype(float)
        label = np.load("../input/" + self.labels[idx]).astype(float)
        label_cls = 1 if label.sum() > 0 else 0
        
        if hasattr(self.cfg.dataset, "pretrain") and self.cfg.dataset.pretrain and (label_cls == 1) and self.mode == "train":
            label = np.load("../input/" + self.df['label_many'][idx]).astype(float)
            label = np.mean(label, axis=3)
        else:
            label = np.load("../input/" + self.labels[idx]).astype(float)        
            
        # label_cls = 1 if label.sum() > 0 else 0
        if self.transform :
            data = self.transform(image=image, mask=label)
            image  = data['image']
            aug_label  = data['mask']
            
            image = np.transpose(image, (2, 0, 1))
            label = np.transpose(aug_label, (2, 0, 1))  
   
        cls_label = (torch.tensor(aug_label)>0).float()

        if hasattr(self.cfg.dataset, "normalize") and self.cfg.dataset.normalize:
            image = self.normalize_image(torch.tensor(image))
            return image, torch.tensor(label), cls_label
        else:
            return torch.tensor(image), torch.tensor(label), cls_label
