import torch, torch.nn as nn, numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

do_cos = nn.CosineSimilarity(dim=1)

def cm(out, fp, ranker, loss):
    pred = (out >= 0.5).type(torch.FloatTensor).cuda()
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

    rank_res = ranker.batched_rank(pred, fp)
    cts = [1, 5, 10]
    # strictly less as batched_rank returns number of items STRICTLY greater
    ranks = {f"rank_{allow}": torch.sum(rank_res < allow).item()/len(rank_res) for allow in cts}
    return {
        "ce_loss": loss.item(),
        "cos": torch.mean(cos).item(),
        "active_bits": active.item(),
        "f1": np.mean(f1),
        "precision": np.mean(precision), 
        "recall": np.mean(recall), 
        "accuracy": np.mean(accuracy),
        **ranks
    }

