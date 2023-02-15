import torch, torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle, os, numpy as np
from pathlib import Path

class GenericIndexedDataset(Dataset):
    """
      Generic Dataset with indexing
      Able to choose what features to extract from the dataset and what features
      to ignore
    """

    def __init__(self, direc, features, split="train", len_override = None):
        self.dir = Path(direc)
        assert(split in ["train", "val", "test"])

        self.split_dir = self.dir / split
        assert(os.path.exists(self.split_dir))

        self.features = features
        self.indexed_features = {}

        self.num_data = None

        for feature in features:
            if os.path.exists(self.split_dir / feature / "index.pkl"):
                with open(self.split_dir / feature / "index.pkl", "rb") as f:
                   feats = pickle.load(f)
                   self.indexed_features[feature] = feats
                   if self.num_data is None:
                       self.num_data = len(feats)
            else:
                if not self.num_data:
                    self.num_data = len(os.listdir(self.split_dir / feature))

        if self.num_data is None:
            raise Exception(f"[{str(self.__class__)}]: Could not interpolate dataset size!!!")
        
        self.len_override = len_override
 
    def __len__(self):
        return self.num_data if self.len_override is None else self.len_override

    def __getitem__(self, idx):
        feats = [None] * len(self.features)
        for idx2, feature in enumerate(self.features):
            if feature not in self.indexed_features:
                feats[idx2] = torch.load(self.split_dir / feature / f"{idx}.pt")
            else:
                feats[idx2] = self.indexed_features[feature][idx]
        return feats

def build_collate_fn(feature_handlers):
    def collate(batch):
      feat_columns = tuple(zip(*batch))
      out = [None] * len(feat_columns)
      for idx, (handler, feat_column) in enumerate(zip(feature_handlers, feat_columns)):
          if handler is None:
              out[idx] = torch.stack(feat_column, dim=0)
          else:
              out[idx] = handler(feat_column)
      return out
    return collate

class GenericIndexedModule(pl.LightningDataModule):
    def __init__(self, dir, features, feature_handlers, batch_size: int = 32, num_workers=4, len_override=None):
        super().__init__()
        self.dir = dir
        self.features = features
        self.feature_handlers = feature_handlers
        self.batch_size = batch_size
        self.collate_fn = build_collate_fn(feature_handlers)
        self.num_workers = num_workers
        self.len_override = len_override
    
    def setup(self, stage):
        if stage == "fit" or stage is None:
            self.train = GenericIndexedDataset(self.dir, self.features, split="train", len_override=self.len_override)
            self.val = GenericIndexedDataset(self.dir, self.features, split="val", len_override=self.len_override)
        if stage == "test":
            self.test = GenericIndexedDataset(self.dir, self.features, split="test", len_override=self.len_override)
        if stage == "predict":
            raise NotImplementedError("Predict setup not implemented")

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=self.num_workers)
