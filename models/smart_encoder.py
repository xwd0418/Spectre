import pytorch_lightning as pl
import logging
import torch
import torch.nn as nn
from models.encoders.encoder_factory import build_encoder, build_encoder_from_args

from models.ranked_transformer import RANKED_TNSFMER_ARGS

SMART_ENCODER_ARGS = [
  "dim_model",
  "dim_coords",
  "heads",
  "layers",
  "ff_dim",
  "coord_enc",
  "enc_args",
  "r_dropout"
]

class SMART_Encoder(pl.LightningModule):
  """A Transformer encoder for input HSQC.
  Parameters
  ----------
  """

  def __init__(
      self,
      # model args
      dim_model=128,
      dim_coords=[43, 43, 42],
      heads=8,
      layers=8,
      ff_dim=1024,
      coord_enc="ce",
      enc_args=None,
      r_dropout=0,
      *args,
      **kwargs,
  ):
    super().__init__()
    params = locals().copy()
    self.out_logger = logging.getLogger("lightning")
    self.out_logger.info("Started Initializing")

    for k, v in params.items():
      if k in SMART_ENCODER_ARGS:
        self.out_logger.info(f"\tHyperparameter: {k}={v}")

    self.save_hyperparameters(*SMART_ENCODER_ARGS)

    # ranked encoder
    self.enc = build_encoder_from_args(
        coord_enc, dim_model, dim_coords, enc_args)
    self.out_logger.info(f"Using {str(self.enc.__class__)}")

    self.dim_model = dim_model

    # === All Parameters ===
    # (1, 1, dim_model)
    self.latent = torch.nn.Parameter(torch.randn(1, 1, dim_model))

    # The Transformer layers:
    layer = torch.nn.TransformerEncoderLayer(
        d_model=dim_model,
        nhead=heads,
        dim_feedforward=ff_dim,
        batch_first=True,
        dropout=r_dropout,
    )
    self.transformer_encoder = torch.nn.TransformerEncoder(
        layer,
        num_layers=layers,
    )
    # === END Parameters ===

  def encode(self, hsqc, mask):
    # (b_s, seq_len, model_dim)
    points = self.enc(hsqc)
    # Add the spectrum representation to each input:
    latent = self.latent.expand(points.shape[0], -1, -1).to(self.device)
    points = torch.cat([latent, points], dim=1)
    # (b_s, seq_len + 1)
    mask = torch.cat([torch.ones(len(mask), 1).to(self.device), mask], dim=1)
    out = self.transformer_encoder(points, src_key_padding_mask=mask)
    return out

  def forward(self, batch) -> torch.Tensor:
    hsqc, mask = batch
    out = self.encode(hsqc, mask)
    return out[:, 0, :]
