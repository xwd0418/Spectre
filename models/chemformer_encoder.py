
# This code is pulled straight from the Chemformer repo
#
import math
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import OneCycleLR
from functools import partial
from textwrap import indent

from models.chemformer.molbart_utils import (
    PreNormEncoderLayer,
    PreNormDecoderLayer,
    FuncLR
)

from models.extras.transformer_stuff import positional_embs


class _Yoinked_Encoder(pl.LightningModule):
  def __init__(
      self,
      pad_token_idx,
      vocab_size,
      d_model,
      num_layers,
      num_heads,
      d_feedforward,
      activation,
      max_seq_len,
      dropout=0.1,
      **kwargs
  ):
    super().__init__()

    self.pad_token_idx = pad_token_idx
    self.vocab_size = vocab_size
    self.d_model = d_model
    self.num_layers = num_layers
    self.num_heads = num_heads
    self.d_feedforward = d_feedforward
    self.activation = activation
    self.max_seq_len = max_seq_len
    self.dropout = dropout

    # Additional args passed in to **kwargs in init will also be saved
    self.save_hyperparameters()

    self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_idx)
    self.dropout = nn.Dropout(dropout)
    self.register_buffer("pos_emb", positional_embs(d_model, max_seq_len))

  def forward(self, x):
    raise NotImplementedError()

  def _calc_loss(self, batch_input, model_output):
    """ Calculate the loss for the model

    Args:
        batch_input (dict): Input given to model,
        model_output (dict): Output from model

    Returns:
        loss (singleton tensor)
    """
    raise NotImplementedError()

  def sample_molecules(self, batch_input, sampling_alg="greedy"):
    """ Sample molecules from the model

    Args:
        batch_input (dict): Input given to model
        sampling_alg (str): Algorithm to use to sample SMILES strings from model

    Returns:
        ([[str]], [[float]]): Tuple of molecule SMILES strings and log lhs (outer dimension is batch)
    """

    raise NotImplementedError()

  def configure_optimizers(self):
    optim = torch.optim.Adam(self.parameters(), lr=self.lr)
    return optim

  def _construct_input(self, token_ids):
    seq_len, _ = tuple(token_ids.size())
    token_embs = self.emb(token_ids)

    # Scaling the embeddings like this is done in other transformer libraries
    token_embs = token_embs * math.sqrt(self.d_model)

    positional_embs = self.pos_emb[:seq_len, :].unsqueeze(0).transpose(0, 1)
    embs = token_embs + positional_embs
    embs = self.dropout(embs)
    return embs

# ----------------------------------------------------------------------------------------------------------
# -------------------------------------------- Pre-train Models --------------------------------------------
# ----------------------------------------------------------------------------------------------------------


class BART_Encoder(_Yoinked_Encoder):

  def __init__(
      self,
      pad_token_idx,
      vocab_size,
      d_model,
      num_layers,
      num_heads,
      d_feedforward,
      activation,
      num_steps,
      max_seq_len,
      dropout=0.1,
      **kwargs
  ):
    super().__init__(
        pad_token_idx,
        vocab_size,
        d_model,
        num_layers,
        num_heads,
        d_feedforward,
        activation,
        max_seq_len,
        dropout,
        **kwargs
    )
    self.num_beams = 10

    enc_norm = nn.LayerNorm(d_model)
    enc_layer = PreNormEncoderLayer(
        d_model, num_heads, d_feedforward, dropout, activation)
    self.encoder = nn.TransformerEncoder(enc_layer, num_layers, norm=enc_norm)

  def forward(self, batch) -> torch.Tensor:
    smiles_token_ids, smiles_pad_mask = batch
    encoder_embs = self._construct_input(smiles_token_ids)
    memory = self.encoder(encoder_embs.permute((1, 0, 2)), src_key_padding_mask=smiles_pad_mask)
    return memory[0, :, :]
