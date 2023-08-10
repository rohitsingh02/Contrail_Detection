
import os, sys, gc 
import torch
import torchvision.transforms as T
import pandas as pd
import numpy as np


###########
class ContrailsDatasetUTAE(torch.utils.data.Dataset):
    def __init__(self, df, cfg, transform=None, mode="train", time_slices = [0,4], vb=False):  
        train = True if mode == "train" else False
        self.df = df
        self.cfg = cfg
        self.trn = train
        self.transform = transform
        self.dir = f'{cfg.data_path}/train' if train else f'{cfg.data_path}/validation'
        self.df_idx: pd.DataFrame = pd.DataFrame({'idx': os.listdir(self.dir)})
        self.normalize_image = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.image_size = cfg.size
        self.time_slices = time_slices ## start/end time slice 
        if mode == "test": self.dir = f'{cfg.data_path}/test'

    
    def read_record(self, directory):
        record_data = {}
        for x in [
            "band_11", 
            "band_14", 
            "band_15"
        ]:
            record_data[x] = np.load(os.path.join(directory, x + ".npy"))
        return record_data
    
    def read_labels(self, directory):
        x = 'human_pixel_masks'
        labels = np.load(os.path.join(directory, x + ".npy"))
        return labels

    def normalize_range(self, data, bounds):
        """Maps data to the range [0, 1]."""
        return (data - bounds[0]) / (bounds[1] - bounds[0])
    
    def get_false_color(self, record_data):
        _T11_BOUNDS = (243, 303)
        _CLOUD_TOP_TDIFF_BOUNDS = (-4, 5)
        _TDIFF_BOUNDS = (-4, 2)
        N_TIMES_BEFORE = 4 #
        T_SLICE_START = self.time_slices[0] #0 #4
        T_SLICE_END = self.time_slices[1] #4 # 6

        r = self.normalize_range(record_data["band_15"] - record_data["band_14"], _TDIFF_BOUNDS)
        g = self.normalize_range(record_data["band_14"] - record_data["band_11"], _CLOUD_TOP_TDIFF_BOUNDS)
        b = self.normalize_range(record_data["band_14"], _T11_BOUNDS)
        false_color = np.clip(np.stack([r, g, b], axis=2), 0, 1)
        #print('false_color', false_color.shape) # 256 x 256 x 3 x 8
        
        # img = false_color[..., :N_TIMES_BEFORE+1]
        img = false_color[..., T_SLICE_START:T_SLICE_END+1]
        #print('false_color', img.shape) ## 256 x 256 x 3 x 4
        return img
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        con_path = row.path
        data = self.read_record(con_path)    
        img = self.get_false_color(data)
        #print(img.shape) ## [3, 4, 256, 256] C x T x H x W

        ############## load mask 
        mask = self.read_labels(con_path) 
        mask = torch.tensor(mask).permute(2, 0, 1)

        # ##############
        # if self.transform:
        #     data = self.transform(image=image, mask=mask)
        #     image  = data['image']
        #     aug_label  = data['mask']
            
        #     image = np.transpose(image, (2, 0, 1))
        #     label = np.transpose(aug_label, (2, 0, 1))  
        ##############
        #img = torch.tensor(np.reshape(img, (CFG.size, CFG.size, 3))).to(torch.float32).permute(2, 0, 1)
        #img = torch.tensor(img).to(torch.float32).permute(2, 3, 0, 1)
         
        img = torch.tensor(img).to(torch.float32).permute(3, 2, 0, 1) ## T x C x H x W 
        
        # img2 = torch.zeros(img.shape)
        # for t in range(img.shape[1]):
        #     img2[t, :, :, :] = self.normalize_image(img[t, :, :, :])
        # #print(img2.shape)
        # img = img2
        
        # del img; gc.collect() 
        image_id = int(self.df_idx.iloc[index]['idx'])
        
        #####
        cls_label = (torch.tensor(mask)>0).float()

        return img.float(), mask.float(), cls_label #torch.tensor(image_id)

########### run 
# ds = ContrailsDatasetUTAE(train_df, CFG, None, mode='train', time_slices=[2,5])
# ds[0][0].shape, ds[0][1].shape
## (torch.Size([4, 3, 256, 256]), torch.Size([1, 256, 256]))
