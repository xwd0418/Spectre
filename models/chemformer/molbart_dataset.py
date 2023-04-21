import random
import functools
import torch
import pandas as pd
import pytorch_lightning as pl
from rdkit import Chem
from pathlib import Path
from typing import Optional
from torch.utils.data import Dataset
from pysmilesutils.augment import MolAugmenter

class _AbsDataset(Dataset):
  def __len__(self):
    raise NotImplementedError()

  def __getitem__(self, item):
    raise NotImplementedError()

  def split_idxs(self, val_idxs, test_idxs):
    raise NotImplementedError()

  def split(self, val_perc=0.2, test_perc=0.2):
    """ Split the dataset randomly into three datasets

    Splits the dataset into train, validation and test datasets.
    Validation and test dataset have round(len * <val/test>_perc) elements in each
    """

    split_perc = val_perc + test_perc
    if split_perc > 1:
      msg = f"Percentage of dataset to split must not be greater than 1, got {split_perc}"
      raise ValueError(msg)

    dataset_len = len(self)
    val_len = round(dataset_len * val_perc)
    test_len = round(dataset_len * test_perc)

    val_idxs = random.sample(range(dataset_len), val_len)
    test_idxs = random.sample(range(dataset_len), test_len)

    train_dataset, val_dataset, test_dataset = self.split_idxs(
        val_idxs, test_idxs)

    return train_dataset, val_dataset, test_dataset

class ReactionDataset(_AbsDataset):
  def __init__(self, reactants, products, items=None, transform=None, aug_prob=0.0):
    super(ReactionDataset, self).__init__()

    if len(reactants) != len(products):
      raise ValueError(
          f"There must be an equal number of reactants and products")

    self.reactants = reactants
    self.products = products
    self.items = items
    self.transform = transform
    self.aug_prob = aug_prob
    self.aug = MolAugmenter()

  def __len__(self):
    return len(self.reactants)

  def __getitem__(self, item):
    reactant = self.reactants[item]
    product = self.products[item]
    output = (reactant, product, self.items[item]) if self.items is not None else (
        reactant, product)
    output = self.transform(*output) if self.transform is not None else output
    return output

  def split_idxs(self, val_idxs, test_idxs):
    """ Splits dataset into train, val and test

    Note: Assumes all remaining indices outside of val_idxs and test_idxs are for training data
    The datasets are returned as ReactionDataset objects, if these should be a subclass 
    the from_reaction_pairs function should be overidden

    Args:
        val_idxs (List[int]): Indices for validation data
        test_idxs (List[int]): Indices for test data

    Returns:
        (ReactionDataset, ReactionDataset, ReactionDataset): Train, val and test datasets
    """

    # Use aug prob of 0.0 for val and test datasets
    val_reacts = [self.reactants[idx] for idx in val_idxs]
    val_prods = [self.products[idx] for idx in val_idxs]
    val_extra = [self.items[idx]
                 for idx in val_idxs] if self.items is not None else None
    val_dataset = ReactionDataset(
        val_reacts, val_prods, val_extra, self.transform, 0.0)

    test_reacts = [self.reactants[idx] for idx in test_idxs]
    test_prods = [self.products[idx] for idx in test_idxs]
    test_extra = [self.items[idx]
                  for idx in test_idxs] if self.items is not None else None
    test_dataset = ReactionDataset(
        test_reacts, test_prods, test_extra, self.transform, 0.0)

    train_idxs = set(range(len(self))) - set(val_idxs).union(set(test_idxs))
    train_reacts = [self.reactants[idx] for idx in train_idxs]
    train_prods = [self.products[idx] for idx in train_idxs]
    train_extra = [self.items[idx]
                   for idx in train_idxs] if self.items is not None else None
    train_dataset = ReactionDataset(
        train_reacts, train_prods, train_extra, self.transform, self.aug_prob)

    return train_dataset, val_dataset, test_dataset

  def _save_idxs(self, df):
    train_idxs = df.index[df["set"] == "train"]
    val_idxs = df.index[df["set"] == "valid"].tolist()
    test_idxs = df.index[df["set"] == "test"].tolist()

    if len(set(val_idxs).intersection(set(test_idxs))) > 0:
      raise ValueError(f"Val idxs and test idxs overlap")
    if len(set(train_idxs).intersection(set(test_idxs))) > 0:
      raise ValueError(f"Train idxs and test idxs overlap")
    if len(set(train_idxs).intersection(set(val_idxs))) > 0:
      raise ValueError(f"Train idxs and val idxs overlap")

    return train_idxs, val_idxs, test_idxs

  def _augment_to_smiles(self, mol, other_mol=None):
    aug = random.random() < self.aug_prob
    mol_aug = self.aug([mol])[0] if aug else mol
    mol_str = Chem.MolToSmiles(mol_aug, canonical=not aug)
    if other_mol is not None:
      other_mol_aug = self.aug([other_mol])[0] if aug else other_mol
      other_mol_str = Chem.MolToSmiles(other_mol_aug, canonical=not aug)
      return mol_str, other_mol_str

    return mol_str

class Uspto50(ReactionDataset):
  def __init__(self, data_path, aug_prob, type_token=False, forward=True):
    path = Path(data_path)
    df = pd.read_pickle(path)
    reactants = df["reactants_mol"].tolist()
    products = df["products_mol"].tolist()
    type_tokens = df["reaction_type"].tolist()

    print("Uspto50 __init()__: ")
    print(f"[DS] {type(df)} {len(df)}")

    super().__init__(reactants, products, items=type_tokens,
                     transform=self._prepare_strings, aug_prob=aug_prob)

    self.type_token = type_token
    self.forward = forward
    self.train_idxs, self.val_idxs, self.test_idxs = self._save_idxs(df)

  def _prepare_strings(self, react, prod, type_token):
    react_str = self._augment_to_smiles(react)
    prod_str = self._augment_to_smiles(prod)

    if self.forward:
      react_str = f"{str(type_token)}{react_str}" if self.type_token else react_str
    else:
      prod_str = f"{str(type_token)}{prod_str}" if self.type_token else prod_str

    return react_str, prod_str
