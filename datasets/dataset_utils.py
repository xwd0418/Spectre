import torch
from torch.nn.utils.rnn import pad_sequence

import sys, pathlib
repo_path = pathlib.Path(__file__).resolve().parents[1]
DATASET_root_path = pathlib.Path("/workspace/")
from utils.matmul_precision_wrapper import set_float32_highest_precision
from rdkit import Chem

from datasets.fp_loader_utils import FP_Loader_Configer, fp_loader_configer

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
  padding_mask = ~torch.any((sequence > 0), dim=2)
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



# fp_loader = None

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot_NMR(hsqc, c_tensor, h_tensor):
    # print(hsqc, c_tensor, h_tensor)
    # Create a 2x2 grid for subplots
    fig = plt.figure(figsize=(6, 4.8))  # Overall figure size
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 20], width_ratios=[1, 20])

    # Create subplots in different locations and sizes
    ax1 = fig.add_subplot(gs[1, 1])  # Takes up the first row
    if hsqc is not None:
        pos = hsqc[hsqc[:,2]>0]
        neg = hsqc[hsqc[:,2]<0]
        ax1.scatter(pos[:,1], pos[:,0], c="blue", label="CH or CH3", s=5)
        ax1.scatter(neg[:,1], neg[:,0], c="red", label="CH2", s=5)
        # print("scatter!!")
        # print(pos, neg)
    ax1.set_title("HSQC")
    ax1.set_xlabel('Proton Shift (1H)')  # X-axis label
    ax1.set_xlim([0, 12])
    ax1.set_ylim([0, 220])
    ax1.invert_yaxis()
    ax1.invert_xaxis()
    ax1.legend()


    ax2 = fig.add_subplot(gs[1, 0])  # Smaller subplot
    if c_tensor is not None:
        ax2.scatter( torch.ones(len(c_tensor)), c_tensor, c="black", s=2)
    ax2.set_ylim([0, 220])
    ax2.set_title("13C-NMR")
    ax2.set_ylabel('Carbon Shift (13C)')
    ax2.set_xticks([])
    ax2.invert_yaxis()
    ax2.invert_xaxis()

    ax3 = fig.add_subplot(gs[0, 1])  # Smaller subplot
    if h_tensor is not None:
        ax3.scatter(h_tensor, torch.ones(len(h_tensor)),c="black", s=2)
    ax3.set_xlim([0, 12])
    ax3.set_title("1H-NMR")
    ax3.set_yticks([])
    ax3.invert_yaxis()
    ax3.invert_xaxis()

    # Adjust layout to prevent overlapping
    plt.tight_layout()

    # Show the plot
    plt.show()


                
def isomeric_to_canonical_smiles(isomeric_smiles):
    try:
        mol = Chem.MolFromSmiles(isomeric_smiles)
        Chem.RemoveStereochemistry( mol ) 
    except:
        # print(isomeric_smiles)
        return None

    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
    
    return canonical_smiles