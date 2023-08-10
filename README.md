# Contrail_Detection 11th Place (Partial Solution)

It's 11th place solution to Kaggle competition: https://www.kaggle.com/competitions/google-research-identify-contrails-reduce-global-warming

### HARDWARE: (The following specs were used to create the original solution)

Almost all models were trained on 2xA30 machine.

* OS: Ubuntu 20.04.4 LTS
* CPU: Intel Xeon Gold 5315Y @3.2 GHz, 8 cores
* RAM: 44Gi 
* GPU: 2 x NVIDIA RTX A30 (24 GB)


### SOFTWARE (python packages are detailed separately in `requirements.txt`):

* Python 3.9.13
* CUDA 11.6
* nvidia drivers v510.73.05

### Training (Rohit Part)

* Downlaod additional training data from https://www.kaggle.com/competitions/google-research-identify-contrails-reduce-global-warming/data and extract it to `./input` directory. and from https://www.kaggle.com/datasets/rohitsingh9990/contrail-utils and extract it to `./input/data_utils` directory.

> Files Description - 
  * `./input/data_utils/train_5_folds.csv` - folds csv file containing image and label paths with other key info.
  * `./input/data_utils/val_df_filled.csv` - this file contain val folder image and label pths.


#### Steps to run training: 
1. From folder src_unet_seg run `./train_swa.sh`
