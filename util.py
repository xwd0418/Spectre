import pytorch_lightning as pl
import torch
import numpy as np
import random
import logging
import os
import sys
from typing import Literal
from config import BaseConfig
from datasets.fp_loader_utils import Hash_Entropy_FP_loader, DB_Specific_FP_loader, Specific_Radius_MFP_loader

_fp_map = {
    "hash_entropy": Hash_Entropy_FP_loader,
    "db_specific_fp": DB_Specific_FP_loader,
    "mfp_specific_radius": Specific_Radius_MFP_loader
}

def seed_everything(seed):
    pl.seed_everything(seed, workers=True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed)
    random.seed(seed)

def init_logger(logger_path):
    logger = logging.getLogger("lightning")
    logger.setLevel(logging.DEBUG)
    file_path = os.path.join(logger_path, "logs.txt")
    os.makedirs(logger_path, exist_ok=True)
    with open(file_path, 'w') as fp: pass
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler(file_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger

def get_fp_loader(base_config: BaseConfig):
    # TODO: original code would update args radius on db_specific_fp, why?
    if base_config.fp_choice == 'mfp_specific_radius':
        raise NotImplementedError() # TODO: figure out conflict with mfp and pick_entropy
    loader = _fp_map[base_config.fp_choice]()
    loader.setup(out_dim = base_config.out_dim, max_radius = base_config.fp_radius)
    return loader