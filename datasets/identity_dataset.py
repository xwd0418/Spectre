import torch
import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader

class IdentityDataset(Dataset):
  def __init__(self):
    super().__init__()
    self.x = 5 * torch.rand(size=(1024, 5)).float()
    self.y = self.x @ torch.tensor([[5, 3, -4, 2, 3]]).float().t() + \
        torch.rand(size=(1024, 1))

  def __len__(self):
    return len(self.x)

  def __getitem__(self, idx):
    return self.x[idx], self.y[idx]

class IdentityModule(pl.LightningDataModule):
  def __init__(self, batch_size: int = 32, num_workers=4):
    super().__init__()
    self.batch_size = batch_size
    self.num_workers = num_workers

  def setup(self, stage):
    if stage == "fit" or stage is None:
      self.train = IdentityDataset()
      self.val = IdentityDataset()
    if stage == "test":
      self.test = IdentityDataset()
    if stage == "predict":
      raise NotImplementedError("Predict setup not implemented")

  def train_dataloader(self):
    return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers)

  def val_dataloader(self):
    return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers)

  def test_dataloader(self):
    return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers)
