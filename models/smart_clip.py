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

    self.ce1 = nn.CrossEntropyLoss()
    self.ce2 = nn.CrossEntropyLoss()

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

  def training_step(self, batch, batch_idx):
    logits_hsqc, logits_smiles = self.forward(batch)
    loss = self._compute_loss(logits_hsqc, logits_smiles)
    self.log("tr/loss", loss)
    self.log("tr/logit_scale", self.logit_scale[0].item())
    metrics = {
        "loss": loss.item(),
        "logit_scale": self.logit_scale[0].item()
    }
    self.training_step_outputs.append(metrics)
    return loss

  def validation_step(self, batch):
    logits_hsqc, logits_smiles = self.forward(batch)
    loss = self._compute_loss(logits_hsqc, logits_smiles)
    metrics = {
        "loss": loss
    }
    self.validation_step_outputs.append(metrics)

  def on_train_epoch_end(self):
    if self.training_step_outputs:
      feats = self.training_step_outputs[0].keys()
      di = {}
      for feat in feats:
        di[f"tr/mean_{feat}"] = np.mean([v[feat]
                                        for v in self.training_step_outputs])
      for k, v in di.items():
        self.log(k, v, on_epoch=True)
      self.training_step_outputs.clear()

  def on_validation_epoch_end(self):
    if self.validation_step_outputs:
      feats = self.validation_step_outputs[0].keys()
      di = {}
      for feat in feats:
        di[f"val/mean_{feat}"] = np.mean([v[feat]
                                          for v in self.validation_step_outputs])
      for k, v in di.items():
        self.log(k, v, on_epoch=True)
      self.validation_step_outputs.clear()

  def _compute_loss(self, logits_hsqc, logits_smiles):
    ground_truth = torch.arange(
        len(logits_hsqc), dtype=torch.long, device=self.device)
    loss = (self.ce1(logits_hsqc, ground_truth) +
            self.ce2(logits_smiles, ground_truth)) / 2
    return loss

  def configure_optimizers(self):
    if not self.scheduler:
      return torch.optim.Adam(self.parameters(), lr=self.lr)
