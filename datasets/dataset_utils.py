import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


def pad(sequence):
  """
    Assume sequence is a (batch)list of sequences that can be variable size.

    Returns {
      sequence: tensor of padded sequence
      padding_mask: binary tensor marking padding
    }
  """
  sequence = pad_sequence([
      torch.tensor(v, dtype=torch.float32) if type(v) is not torch.Tensor
      else v.type(torch.float32)
      for v in sequence
  ], batch_first=True)
  return sequence

def pad_and_mask(sequence):
  """
    Assume sequence is a (batch)list of sequences that can be variable size.

    Returns {
      sequence: tensor of padded sequence
      padding_mask: binary tensor marking padding
    }
  """
  sequence = pad_sequence([
      torch.tensor(v, dtype=torch.float32) if type(v) is not torch.Tensor
      else v.type(torch.float32)
      for v in sequence
  ], batch_first=True)
  padding_mask = ~(sequence > 0)
  return {
      "sequence": sequence,
      "padding_mask": padding_mask
  }

def tokenise_and_mask(smiles, tokeniser):
  """
    Assume sequence is [batch] sized list of SMILES strings

    Returns {
      raw_smiles: the original list of smiles passed in
      target: right-shifted padded list of token ids (b_s, seq_len)
      target_mask: right-shifted padding mask
      decoder_inputs: tokenised inputs (minus end(&) token)
      decoder_mask: padding mask
    }
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

def tokenise_and_mask_encoder(smiles, tokeniser):
  """
    Assume sequence is [batch] sized list of SMILES strings

    Returns {
      raw_smiles: the original list of smiles passed in
      target: right-shifted padded list of token ids (b_s, seq_len)
      target_mask: right-shifted padding mask
      decoder_inputs: tokenised inputs (minus end(&) token)
      decoder_mask: padding mask
    }
  """

  obj = tokeniser.tokenise(smiles, pad=True)

  out_tokens = obj["original_tokens"]
  out_mask = obj["original_pad_masks"]

  token_ids = torch.tensor(
      tokeniser.convert_tokens_to_ids(out_tokens), dtype=torch.int64)
  padding_mask = torch.tensor(out_mask, dtype=torch.bool)

  # (b_s, seq_len)
  decoder_inputs = token_ids[:, :-1]
  decoder_mask = padding_mask[:, :-1]
  return {
      "raw_smiles": smiles,
      "encoder_inputs": decoder_inputs,
      "encoder_mask": decoder_mask
  }
