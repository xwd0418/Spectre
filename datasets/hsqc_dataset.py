import torch, os, torch.nn as nn, pytorch_lightning as pl, numpy as np
from torch.utils.data import DataLoader, Dataset
import logging

class HSQCDataset(Dataset):
    def __init__(self, split="train"):
        self.dir = "/workspace/smart4.5/tempdata"
        self.split = split
        self.orig_hsqc = os.path.join(self.dir, "data")
        assert(os.path.exists(self.orig_hsqc))
        assert(split in ["train", "val", "test"])
        self.files = list(os.listdir(os.path.join(self.orig_hsqc, split, "HSQC_2ch")))
    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        hsqc = np.load(os.path.join(self.dir, "data", self.split, "HSQC_2ch", self.files[i]))
        mfp = np.load(os.path.join(self.dir, "data", self.split, "fingerprint", self.files[i]))
        return hsqc.astype(np.float32), mfp.astype(np.float32)

class HSQCDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage):
        if stage == "fit" or stage is None:
            self.train = HSQCDataset(split="train")
            self.val = HSQCDataset(split="val")
        if stage == "test":
            raise NotImplementedError("Test setup not implemented")
        if stage == "predict":
            raise NotImplementedError("Predict setup not implemented")

    def train_dataloader(self):
        return DataLoader(self.train, shuffle=True, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val, shuffle=True, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        raise NotImplementedError("Test dataloader not implemented")

    def predict_dataloader(self):
        raise NotImplementedError("Predict dataloader not implemented")

    def teardown(self, stage):
        pass
