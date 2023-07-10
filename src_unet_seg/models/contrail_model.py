import torch.nn as nn
import segmentation_models_pytorch as smp
import torch
from torch import nn, optim
import torch.nn.functional as F
import pretrainedmodels
import timm

class Net(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        default_weight = 'imagenet'
        if hasattr(cfg.architecture, 'default_weight'):
            default_weight = cfg.architecture.default_weight
        
        self.model = smp.Unet(
            encoder_name=cfg.architecture.backbone,      # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=default_weight,     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=cfg.dataset.num_classes,        # model output channels (number of classes in your dataset)
            activation=None
        )
        
    
    def forward(self, inputs):
        mask = self.model(inputs)
        return mask


# ### UNET_SERESNEXT101




### TimmUnet model | Model2 Nirjhar

class TimmSegModel(nn.Module):
    def __init__(self, cfg, segtype='unet', pretrained=True):
        super(TimmSegModel, self).__init__()

        self.n_blocks = 4
        self.encoder = timm.create_model(
            cfg.architecture.backbone,
            in_chans=3,
            features_only=True,
            drop_rate=0.5,
            pretrained=True
        )
        g = self.encoder(torch.rand(1, 3, 128, 128))
        encoder_channels = [1] + [_.shape[1] for _ in g]
        decoder_channels = [256, 128, 64, 32, 16]
        if segtype == 'unet':
            self.decoder = smp.decoders.unetplusplus.decoder.UnetPlusPlusDecoder(
                encoder_channels=encoder_channels[:self.n_blocks+1],
                decoder_channels=decoder_channels[:self.n_blocks],
                n_blocks=self.n_blocks,
            )

        self.segmentation_head = nn.Sequential(
            nn.Conv2d(decoder_channels[self.n_blocks-1], 
                      cfg.dataset.num_classes,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=(1, 1)
            ),
            nn.UpsamplingBilinear2d(scale_factor=1)
        )

    def forward(self,x):
        global_features = [0] + self.encoder(x)[:self.n_blocks]
        seg_features = self.decoder(*global_features)
        seg_features = self.segmentation_head(seg_features)
        return seg_features




