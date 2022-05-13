import torch, torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import os 
import numpy as np

class MsDataset(Dataset):
    """
    Dataset for pretraining with MS information only:
    depends on colin-devpod file configuration
    TODO: train/val split
    """

    def __init__(self, split="train"):
        self.dir = "/workspace/data"
        assert(split in ["train", "val", "test"])
        self.split = os.path.join(self.dir, split)
        self.labels = os.path.join(self.split, "r3fingerprints")
        self.ms = os.path.join(self.split, "spectra")
        assert(os.path.exists(self.labels) and os.path.exists(self.ms))
        self.ids = list(os.listdir(self.ms))
    
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        ccmslid = self.ids[idx]
        spectra = np.load(os.path.join(self.ms, ccmslid))
        fingerprint = np.load(os.path.join(self.labels, ccmslid))
        return spectra, fingerprint, ccmslid

def pad(batch):
    ms, fp, ccmslid = zip(*batch)
    fp = np.array(fp)
    ms = pad_sequence([torch.tensor(v, dtype=torch.float) for v in ms], batch_first=True)
    return ms, torch.tensor(fp, dtype=torch.float), ccmslid

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
