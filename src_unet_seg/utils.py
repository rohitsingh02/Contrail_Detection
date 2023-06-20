import numpy as np 
import torch 
import random
import os
from sklearn.model_selection import StratifiedKFold
import segmentation_models_pytorch as smp
from imblearn.under_sampling import RandomUnderSampler


def seed_everything(seed=42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    
    
def balance_target(df, target_column, balance_ratio=1.0):
    """
    Balance the target variable in a dataframe by performing undersampling.
    
    Args:
        df (pandas.DataFrame): The input dataframe.
        target_column (str): The name of the target column.
        balance_ratio (float, optional): The desired ratio of minority to majority class.
            Default is 1.0 (no change).
    
    Returns:
        pandas.DataFrame: The balanced dataframe.
    """
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Create the undersampler
    undersampler = RandomUnderSampler(sampling_strategy=balance_ratio, random_state=42)
    
    # Perform undersampling
    X_resampled, y_resampled = undersampler.fit_resample(X, y)
    
    # Combine the undersampled data into a balanced dataframe
    balanced_df = X_resampled.copy()
    balanced_df[target_column] = y_resampled
    
    return balanced_df
    
    
def create_stratified_folds(df, cfg):
    skf = StratifiedKFold(n_splits = cfg.dataset.num_folds, shuffle = True, random_state = cfg.environment.seed)    
    for idx, (tr_idx, val_idx) in enumerate(skf.split(df, df["class"])):
        df.loc[val_idx, 'fold'] = int(idx)
    return df
    
    
# ====================================================
# Helper functions
# ====================================================
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        
        
def get_logger(cfg):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    filename=f"{cfg.output_dir}/train"    
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    filename=filename
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


def get_non_empty_masks(df, pos_ids):
    df['id'] = df['image'].apply(lambda x: x.split('/')[-2])
    df = df.loc[df['id'].isin(pos_ids)].reset_index(drop=True)
    return df