import torch, torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle
import os 
import numpy as np

class MsDataset(Dataset):
    """
    Dataset for pretraining with MS information only:
    depends on colin-devpod file configuration
    TODO: train/val split
    """

    def __init__(self, split="train"):
        self.dir = "/workspace/data/ms_pretrain"
        assert(split in ["train", "val", "test"])
        self.split_dir = os.path.join(self.dir, split+".pkl")
        assert(os.path.exists(self.split_dir))
        self.data = pickle.load(open(self.split_dir, "rb"))
        self.ids = list(self.data.keys())
    
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sample_num = self.ids[idx]
        spectra = self.data[sample_num]["MS"]
        fingerprint = self.data[sample_num]["FP"]
        return spectra, fingerprint

def pad(batch):
    ms, fp = zip(*batch)
    ms = pad_sequence([torch.tensor(v, dtype=torch.float) for v in ms], batch_first=True)
    return ms, torch.tensor(np.array(fp), dtype=torch.float)

class MsDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32):
        super().__init__()
        self.batch_size = batch_size
        self.collate_fn = pad
    
    def setup(self, stage):
        if stage == "fit" or stage is None:
            self.train = MsDataset(split="train")
            self.val = MsDataset(split="val")
        if stage == "test":
            self.test = MsDataset(split="test")
        if stage == "predict":
            raise NotImplementedError("Predict setup not implemented")

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=4)
