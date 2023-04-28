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

def tokenise_and_mask(smiles, tokeniser):
  """
    Assume sequence is [batch] sized list of SMILES strings
  """

  obj = tokeniser.tokenise(smiles, pad=True)

  out_tokens = obj["original_tokens"]
  out_mask = obj["original_pad_masks"]

  token_ids = torch.tensor(
      tokeniser.convert_tokens_to_ids(out_tokens), dtype=torch.int64)
  padding_mask = torch.tensor(out_mask, dtype=torch.bool)
  targets = token_ids.clone()[:, 1:]
  target_mask = padding_mask.clone()[:, 1:]
  decoder_inputs = token_ids[:, :-1]
  decoder_mask = padding_mask[:, :-1]
  return {
      "raw_smiles": smiles,
      "target": targets,
      "target_mask": target_mask,
      "decoder_inputs": decoder_inputs,
      "decoder_mask": decoder_mask
  }
