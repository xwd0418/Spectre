import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np
from textwrap import indent

class IdentityModule(pl.LightningModule):
  def __init__(self, lr, **kwargs):
    super().__init__()
    self.save_hyperparameters(*["lr"])
    self.lr = lr
    self.v = nn.Linear(5, 1)
    self.loss = torch.nn.MSELoss()

    self.train_vals = []
    self.val_vals = []

  def forward(self, x):
    return self.v(x)

  def training_step(self, batch, batch_idx):
    x, y = batch
    out = self.forward(x)
    loss = self.loss(out, y)
    self.train_vals.append(loss.item())
    return loss

  def on_train_epoch_end(self):
    mean_loss = np.mean(self.train_vals)
    self.train_vals.clear()
    self.log("tr/mean_loss", mean_loss, on_epoch=True)
    return mean_loss

  def validation_step(self, batch, batch_idx):
    x, y = batch
    out = self.forward(x)
    loss = self.loss(out, y)
    self.val_vals.append(loss.item())

  def on_validation_epoch_end(self):
    mean_loss = np.mean(self.val_vals)
    self.val_vals.clear()
    self.log("val/mean_loss", mean_loss, on_epoch=True)

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=self.lr)
