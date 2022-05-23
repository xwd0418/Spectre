import torch, os, torch.nn as nn, pytorch_lightning as pl, numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import logging

class PairDataset(Dataset):
    def __init__(self, split="train"):
        self.dir = "/workspace/smart4.5/tempdata"
        self.split = split
        self.orig_hsqc = os.path.join(self.dir, "new_split")
        self.new_specs = os.path.join(self.dir, "new_split")
        assert(os.path.exists(self.orig_hsqc) and os.path.exists(self.new_specs))
        assert(split in ["train", "val", "test"])
        self.files = list(os.listdir(os.path.join(self.new_specs, split, "mass_spec_scaled")))
    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        hsqc = np.load(os.path.join(self.dir, "new_split", self.split, "HSQC_2ch", self.files[i]))
        ms = np.load(os.path.join(self.dir, "new_split", self.split, "mass_spec_scaled", self.files[i]))
        mfp = np.load(os.path.join(self.dir, "new_split", self.split, "fingerprint", self.files[i]))
        return hsqc, ms, mfp

def pad(batch):
    hsqc, ms, fp = zip(*batch)
    ms = pad_sequence([torch.tensor(v, dtype=torch.float) for v in ms], batch_first=True)
    return torch.tensor(np.array(hsqc), dtype=torch.float), ms, torch.tensor(np.array(fp), dtype=torch.float)

class PairDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32):
        super().__init__()
        self.batch_size = batch_size
        self.colate_fn = pad

    def setup(self, stage):
        if stage == "fit" or stage is None:
            self.train = PairDataset(split="train")
            self.val = PairDataset(split="val")
        if stage == "test":
            raise NotImplementedError("Test setup not implemented")
        if stage == "predict":
            raise NotImplementedError("Predict setup not implemented")

    def train_dataloader(self):
        return DataLoader(self.train, shuffle=True, batch_size=self.batch_size, collate_fn=pad, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, collate_fn=pad, num_workers=4)

    def test_dataloader(self):
        raise NotImplementedError("Test dataloader not implemented")

    def predict_dataloader(self):
        raise NotImplementedError("Predict dataloader not implemented")

    def teardown(self, stage):
        pass
