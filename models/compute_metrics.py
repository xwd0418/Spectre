import torch, torch.nn as nn, numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

do_cos = nn.CosineSimilarity(dim=1)

def cm(out, fp, ranker, loss, loss_fn, thresh = 0.0, device="cuda"):
    pred = (out >= thresh).type(torch.FloatTensor).to(device)
    pred_cpu = pred.cpu()
    labels = fp.cpu()
    # cos
    cos = do_cos(fp, pred)
    # bit activity
    active = torch.mean(torch.sum(pred, axis=1))
    # bit metrics
    f1 = f1_score(labels.flatten(), pred_cpu.flatten())
    precision = precision_score(labels.flatten(), pred_cpu.flatten())
    recall = recall_score(labels.flatten(), pred_cpu.flatten())
    accuracy = accuracy_score(labels.flatten(), pred_cpu.flatten())

    if thresh == 0.5: # probabiltiies
        pos_contr = torch.where(fp == 0, torch.zeros_like(fp, dtype=torch.float), out)
        neg_contr = torch.where(fp == 1, torch.ones_like(fp, dtype=torch.float), out)
    elif thresh == 0.0: # logits
        pos_contr = torch.where(fp == 0, -999*torch.ones_like(fp, dtype=torch.float), out)
        neg_contr = torch.where(fp == 1, 999*torch.ones_like(fp, dtype=torch.float), out)
    else:
        raise(f"Weird Threshold {thresh}")

    pos_loss = loss_fn(pos_contr, fp)
    neg_loss = loss_fn(neg_contr, fp)

    rank_res = ranker.batched_rank(pred, fp)
    cts = [1, 5, 10]
    # strictly less as batched_rank returns number of items STRICTLY greater
    ranks = {f"rank_{allow}": torch.sum(rank_res < allow).item()/len(rank_res) for allow in cts}
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

