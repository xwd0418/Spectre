import torch
import torch.nn as nn
import numpy as np
from torchmetrics.classification import BinaryRecall, BinaryPrecision, BinaryF1Score

do_cos = nn.CosineSimilarity(dim=1)
do_f1 = BinaryF1Score()
do_recall = BinaryRecall()
do_precision = BinaryPrecision()
do_accuracy = BinaryAccuracy()

def cm(model_output, fp_label, ranker, loss, loss_fn, thresh=0.0, device="cuda"):
  # Fingeprint prediction
  fp_pred = model_output >= thresh

  # cos
  cos = do_cos(fp_label, fp_pred)
  # bit activity
  active = torch.mean(torch.sum(fp_pred, axis=1))
  # bit metrics
  f1 = do_f1(fp_pred, fp_label)
  precision = do_precision(fp_pred, fp_label)
  recall = do_recall(fp_pred, fp_label)
  accuracy = do_accuracy(fp_pred, fp_label)

  if torch.isclose(thresh, 0.5):  # probabiltiies
    pos_contr = torch.where(
        fp_label == 0, torch.zeros_like(fp_label, dtype=torch.float), model_output)
    neg_contr = torch.where(
        fp_label == 1, torch.ones_like(fp_label, dtype=torch.float), model_output)
  elif torch.isclose(thresh, 0.0):  # logits
    pos_contr = torch.where(
        fp_label == 0, -999 * torch.ones_like(fp_label, dtype=torch.float), model_output)
    neg_contr = torch.where(
        fp_label == 1, 999 * torch.ones_like(fp_label, dtype=torch.float), model_output)
  else:
    raise (f"Weird Threshold {thresh}")

  pos_loss = loss_fn(pos_contr, fp_label)
  neg_loss = loss_fn(neg_contr, fp_label)

  # === Do Ranking ===
  rank_res = ranker.batched_rank(fp_pred, fp_label)
  cts = [1, 5, 10]
  # strictly less as batched_rank returns number of items STRICTLY greater
  print(rank_res)
  ranks = {
      f"rank_{allow}": torch.sum(rank_res < allow).item() / torch.numel(rank_res)
      for allow in cts
  }
  mean_rank = torch.mean(rank_res.float()).item()
  return {
      "ce_loss": loss.item(),
      "pos_loss": pos_loss.item(),
      "neg_loss": neg_loss.item(),
      "pos_neg_loss": (pos_loss + neg_loss).item(),
      "cos": torch.mean(cos).item(),
      "active_bits": active.item(),
      "f1": np.mean(f1),
      "precision": np.mean(precision),
      "recall": np.mean(recall),
      "accuracy": np.mean(accuracy),
      "mean_rank": mean_rank,
      **ranks
  }
