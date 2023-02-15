import numpy as np, torch
from torch.nn.utils.rnn import pad_sequence

def pad(sequence):
  """
    Assume sequence is [batch, <whatever>]
  """
  sequence = pad_sequence([torch.tensor(v, dtype=torch.float) if type(v) is not torch.Tensor else v for v in sequence], batch_first=True)
  return sequence