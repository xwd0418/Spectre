import pytorch_lightning as pl
import torch, torch.nn as nn
from old_encoder import MassSpecEncoder
from sklearn.metrics import f1_score  
import numpy as np

class EncoderOnly(pl.LightningModule):
    def __init__(
            self,
            lr = 1e-3,
            dim_model=128,
            out_dim = 6144
        ):
        super().__init__()
        self.lr = lr
        self.enc = MassSpecEncoder(dim_model=dim_model, lr=lr)
        self.fc = nn.Linear(dim_model, out_dim)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss()
    
        self.cos = nn.CosineSimilarity(dim=1)

    def encode(self, x):
        return self.enc(x)

    def forward(self, x):
        out = self.enc(x)
        out = self.fc(out).squeeze(1)
        out = self.sigmoid(out)
        return out

    def training_step(self, batch, batch_idx):
        x, labels, ccmslid = batch
        out = self.forward(x)
        loss = self.loss(out, labels)
        self.log("tr/train", loss)
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, labels, ccmslid = batch
        out = self.forward(x)
        loss = self.loss(out, labels)
        self.log("val/loss", loss)

        predicted = (out >= 0.5)
        # cosine similarity
        cos = self.cos(labels, predicted)
        self.log("val/cosine_sim", torch.mean(cos).item())

        # f1 score
        predicted = predicted.cpu()
        labels = labels.cpu()
        f1 = f1_score(predicted.flatten(), labels.flatten())
        self.log("val/f1", np.mean(f1))
        return loss
    
    def test_step(self, batch, batch_idx):
        x, labels, ccmslid = batch
        out = self.forward(x)
        loss = self.loss(out, labels)
        self.log("test/loss", loss)

        predicted = (out >= 0.5)
        # cosine similarity
        cos = self.cos(labels, predicted)
        self.log("test/cosine_sim", torch.mean(cos).item())

        # f1 score
        predicted = predicted.cpu()
        labels = labels.cpu()
        f1 = np.mean(f1_score(predicted.flatten(), labels.flatten()))
        self.log("test/f1", f1)
        return {"test_loss":loss, "test_f1":f1}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)