import torch
import pandas as pd
import numpy as np
import random
import torchvision.transforms as T



class ContrailDataset:
    def __init__(self, df, cfg, transform=None, mode="train"):
        self.df = df  
        self.cfg = cfg           
        self.images = df['image']
        self.labels = df['label']
        self.transform =transform
        self.mode=mode        

        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        image = np.load("../input/" + self.images[idx]).astype(float)   
        label = np.load("../input/" + self.labels[idx]).astype(float)
            
        # label_cls = 1 if label.sum() > 0 else 0
        if self.transform :
            data = self.transform(image=image, mask=label)
            image  = data['image']
            aug_label  = data['mask']
            
            image = np.transpose(image, (2, 0, 1))
            label = np.transpose(aug_label, (2, 0, 1))  
   
        cls_label = (torch.tensor(aug_label)>0).float()
        return torch.tensor(image), torch.tensor(label), cls_label

    