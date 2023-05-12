import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np

from models.chemformer_encoder import BART_Encoder
from models.smart_encoder import SMART_Encoder

class SMART_CLIP(pl.LightningModule):
  def __init__(self,
               chemformer_path,
               projection_dim,
               # optmizer
               lr=1e-3,
               **kwargs):
    # === Parameters ===
    self.bart = BART_Encoder.load_from_checkpoint(chemformer_path, strict=False)
    self.smart = SMART_Encoder(**kwargs)

    self.projection_dim = projection_dim
    self.bart_lin = nn.Linear(512, projection_dim)
    self.smart_lin = nn.Linear(kwargs["dim_model"], projection_dim)

    self.logit_scale = nn.Parameter(torch.tensor([np.log(1 / 0.07)]))

    # === Optimizer ===
    self.validation_step_outputs = []
    self.training_step_outputs = []
    self.lr = lr
    self.loss_fn = nn.CrossEntropyLoss(reduction="none")

  def encode(self, hsqc_vals, hsqc_mask, smiles_vals, smiles_mask):
    hsqc_out = self.smart(hsqc_vals, hsqc_mask)
    hsqc_proj = self.smart_lin(hsqc_out)
    hsqc_proj = hsqc_proj / hsqc_proj.norm(dim=1, keepdim=True)

    chemformer_out = self.bart(smiles_vals, smiles_mask)
    chemformer_proj = self.bart_lin(chemformer_out)
    chemformer_proj = chemformer_proj / \
        chemformer_proj.norm(dim=1, keepdim=True)

    logit_scale = self.logit_scale.exp()
    logits_per_hsqc = logit_scale * hsqc_proj @ chemformer_proj.t()
    logits_per_smiles = logits_per_hsqc.t()

    return logits_per_hsqc, logits_per_smiles

  def optimizer_step(self, *args, **kwargs):
    super().optimizer_step(*args, **kwargs)
    with torch.no_grad():
      self.logit_scale.clamp_(-100, 100)

  def forward(self, batch):
    hsqc, hsqc_p, smiles, smiles_p = batch
    return self.encode(hsqc, hsqc_p, smiles, smiles_p)

  def configure_optimizers(self):
    if not self.scheduler:
      return torch.optim.Adam(self.parameters(), lr=self.lr)
