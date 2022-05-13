import pytorch_lightning as pl
import torch, torch.nn as nn
from sklearn.metrics import f1_score  
import torchvision.models as models
import numpy as np

class HSQCResnet(pl.LightningModule):
    def __init__(
            self, 
            dim_ff=512,
            out_dim=6144,
            lr = 1e-4
        ):
        super().__init__()
        
        self.proj = nn.Conv2d(2, 3, 1)
        self.features = models.resnet50(pretrained=False)
        self.lr = lr

        self.in_feat = self.features.fc.in_features
        self.features.fc = nn.Sequential()
        self.lin_fp = nn.Sequential(
            nn.Linear(self.in_feat, self.in_feat),
            nn.ReLU(),
            nn.Linear(self.in_feat, 256),
            nn.ReLU(),
            nn.Linear(256, 6144)
        )
        self.loss = nn.BCEWithLogitsLoss()
        self.cos = nn.CosineSimilarity(dim=1)
        
    def forward(self, x):
        hsqc = x
        hsqc = self.proj(torch.permute(hsqc, (0, 3, 1, 2)))
        hsqc_embed = self.features(hsqc)
        feats = hsqc_embed
        out = self.lin_fp(feats)
        return out

    def training_step(self, batch, batch_idx):
        hsqc, labels = batch
        out = self(hsqc)
        loss = self.loss(out, labels)

        self.log("tr/loss", loss)
        return loss

    def training_epoch_end(self, train_step_outputs):
        mean_loss = sum([t["loss"].item() for t in train_step_outputs]) / len(train_step_outputs)
        self.log("tr/mean_loss", mean_loss)

    def validation_step(self, batch, batch_idx):
        hsqc, labels = batch
        out = self(hsqc)
        loss = self.loss(out, labels)
        pred_labels = (out >= 0)
        cos = self.cos(labels, pred_labels)

        pred_labels = pred_labels.cpu()
        labels = labels.cpu()

        f1 = f1_score(labels.flatten(), pred_labels.flatten())
        
        cos_val = torch.mean(cos).item()
        f1_val = np.mean(f1)
        return {"ce_loss": loss.item(), "cos": cos_val, "f1": f1_val}

    def validation_epoch_end(self, validation_step_outputs):
        feats = validation_step_outputs[0].keys()
        di = {}
        for feat in feats:
            di[f"val/mean_{feat}"] = np.mean([v[feat] for v in validation_step_outputs])
        for k,v in di.items():
            self.log(k, v)

    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = self.lr)
