from argparse import ArgumentParser
from typing import Optional
import torch, os, json, math, torch.nn as nn, pytorch_lightning as pl, numpy as np, scipy.stats as stats, random
import logging
from torch.utils.data import DataLoader, Dataset

class DS(Dataset):
    def __init__(self):
        pass
    def __len__(self):
        return 1000

    def __getitem__(self, i):
        t1 = torch.rand(50)
        t2 = torch.tensor(range(1, 51)) * t1 + 1
        return t1, t2


class DSM(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32):
        super().__init__()
        self.batch_size = batch_size
        self.tnsfm = "Asdf"

    def setup(self, stage: Optional[str] = None):
        self.mnist_test = DS()
        self.mnist_predict = DS()
        self.mnist_train, self.mnist_val = DS(), DS()

    def train_dataloader(self):
        return DataLoader(self.mnist_train, num_workers=4, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val,num_workers=4, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, num_workers=4, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, num_workers=4, batch_size=self.batch_size)

    def teardown(self, stage: Optional[str] = None):
        # Used to clean-up when the run is finished
        pass

class Model(pl.LightningModule):
    def __init__(self, hparam1=111, lr = 1e-3, **kwargs):
        super().__init__()
        self.hparams["set"] = 999
        self.lr = lr
        self.save_hyperparameters()
        self.lin = nn.Linear(50, 50)
        self.err = nn.MSELoss()
        self.metric = np.inf

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        parser.add_argument("--model_arg1", type=int, default=12)
        parser.add_argument("--model_arg2", type=str, default="/some/path")
        parser.add_argument("--hparam1", type=int, default=69)
        return parent_parser

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams)

    def forward(self, s):
        return self.lin(s) + self.hparam["hparam1"]

    def training_step(self, batch, batch_idx):
        tr, va = batch
        out = self.lin(tr)
        loss = self.err(out, va)
        self.log("tr/loss", loss)
        return loss

    def training_epoch_end(self, train_step_outputs):
        mean_loss = torch.stack([t["loss"] for t in train_step_outputs]).mean()
        self.log("tr/mean_loss", mean_loss)
    def validation_step(self, batch, batch_idx):
        tr, va = batch
        out = self.lin(tr)
        loss = self.err(out, va)
        return {"val_loss": loss}
    def validation_epoch_end(self, validation_step_outputs):
        mean_loss_v = torch.stack([t["val_loss"] for t in validation_step_outputs]).mean()
        self.log("va/mean_loss", mean_loss_v)
        self.metric = min(self.metric, mean_loss_v.item())
        self.log("hp_metric", self.metric)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
