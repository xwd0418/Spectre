import numpy as np, torch
from torch.nn.utils.rnn import pad_sequence

def pad(sequence):
  """
    Assume sequence is [batch, <whatever>]
  """
  sequence = pad_sequence([torch.tensor(v, dtype=torch.float) for v in sequence], batch_first=True)
  return sequence