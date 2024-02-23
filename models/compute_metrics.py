import torch
import torch.nn as nn
import numpy as np
from torchmetrics.classification import BinaryRecall, BinaryPrecision, BinaryF1Score, BinaryAccuracy

do_cos = nn.CosineSimilarity(dim=1)
do_f1 = BinaryF1Score()
do_recall = BinaryRecall()
do_precision = BinaryPrecision()
do_accuracy = BinaryAccuracy()


def cm(model_output, fp_label, ranker, loss, loss_fn, thresh: float = 0.0, rank_by_soft_output=False, query_idx_in_rankingset=None):
    # return {
    #     "ce_loss": loss.item(),
    # }
    global do_f1, do_recall, do_precision, do_accuracy
    do_f1 = do_f1.to(model_output)
    do_recall = do_recall.to(model_output)
    do_precision = do_precision.to(model_output)
    do_accuracy = do_accuracy.to(model_output)

    # Fingeprint prediction
    fp_pred = (model_output >= thresh).type(torch.cuda.FloatTensor)

    # cos
    cos = torch.mean(do_cos(fp_label, fp_pred)).item()
    # bit activity
    active = torch.mean(torch.sum(fp_pred, axis=1)).item()
    # bit metrics
    f1 = do_f1(fp_pred, fp_label).item()
    prec = do_precision(fp_pred, fp_label).item()
    rec = do_recall(fp_pred, fp_label).item()
    acc = do_accuracy(fp_pred, fp_label).item()

    if np.isclose(thresh, 0.5):  # probabiltiies
        pos_contr = torch.where(
            fp_label == 0, torch.zeros_like(fp_label, dtype=torch.float), model_output)
        neg_contr = torch.where(
            fp_label == 1, torch.ones_like(fp_label, dtype=torch.float), model_output)
    elif np.isclose(thresh, 0.0):  # logits
        pos_contr = torch.where(
            fp_label == 0, -999 * torch.ones_like(fp_label, dtype=torch.float), model_output)
        neg_contr = torch.where(
            fp_label == 1, 999 * torch.ones_like(fp_label, dtype=torch.float), model_output)
    else:
        raise (f"Weird Threshold {thresh}")

    # print( pos_contr.device, fp_label.device)
    pos_loss = loss_fn(pos_contr, fp_label)
    neg_loss = loss_fn(neg_contr, fp_label)

    # === Do Ranking ===
    if rank_by_soft_output:
        rank_res = ranker.batched_rank(torch.sigmoid(model_output), fp_label, query_idx_in_rankingset)
    else:
        rank_res = ranker.batched_rank(fp_pred, fp_label, query_idx_in_rankingset)
    cts = [1, 5, 10]
    # strictly less as batched_rank returns number of items STRICTLY greater
    ranks = {
        f"rank_{allow}": torch.sum(rank_res < allow).item() / torch.numel(rank_res)
        for allow in cts
    }
    # print("ranks", ranks)
    mean_rank = torch.mean(rank_res.float()).item()
    return {
        "ce_loss": loss.item(),
        "pos_loss": pos_loss.item(),
        "neg_loss": neg_loss.item(),
        "pos_neg_loss": (pos_loss + neg_loss).item(),
        "cos": cos,
        "active_bits": active,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "accuracy": acc,
        "mean_rank": mean_rank,
        **ranks
    }
