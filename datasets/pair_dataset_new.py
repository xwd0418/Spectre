import torch, torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle
import os 
import numpy as np

class PairDataset(Dataset):
    """
    Dataset for HSQC and Spectra pairs. HSQC are given as coordinates (not images)
    depends on colin-devpod file configuration
    """

    def __init__(self, split="train"):
        self.dir = "/workspace/data/hsqc_ms_pairs"
        assert(split in ["train", "val", "test"])
        self.split_dir = os.path.join(self.dir, split+".pkl")
        assert(os.path.exists(self.split_dir))
        self.data = pickle.load(open(self.split_dir, "rb"))
        self.ids = list(self.data.keys())
    
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sample_num = self.ids[idx]
        hsqc = self.data[sample_num]["HSQC"]
        spectra = self.data[sample_num]["MS"]
        fingerprint = self.data[sample_num]["FP"]
        return hsqc, spectra, fingerprint

def pad(batch):
    hsqc, ms, fp = zip(*batch)
    hsqc = pad_sequence([torch.tensor(v, dtype=torch.float) for v in hsqc], batch_first=True)
    ms = pad_sequence([torch.tensor(v, dtype=torch.float) for v in ms], batch_first=True)
    return hsqc, ms, torch.tensor(np.array(fp), dtype=torch.float)

class HsqcDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32):
        super().__init__()
        self.batch_size = batch_size
        self.collate_fn = pad
    
    def setup(self, stage):
        if stage == "fit" or stage is None:
            self.train = PairDataset(split="train")
            self.val = PairDataset(split="val")
        if stage == "test":
            self.test = PairDataset(split="test")
        if stage == "predict":
            raise NotImplementedError("Predict setup not implemented")

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=4)
