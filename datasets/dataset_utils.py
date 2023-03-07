import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

def pad(sequence):
  """
    Assume sequence is [batch, <whatever>]
  """
  sequence = pad_sequence([
      torch.tensor(v, dtype=torch.float32) if type(v) is not torch.Tensor
      else v.type(torch.FloatTensor)
      for v in sequence
  ], batch_first=True)

  return sequence
