import pytorch_lightning as pl
import torch, torch.nn as nn, torch.nn.functional as F
from models import SpectraTransformer, HsqcTransformer
from sklearn.metrics import f1_score  
import numpy as np

class baseline_double_transformer(pl.LightningModule):
    """ A baseline implementation of both hsqc and ms transformers
    Parameters
    ----------
    lr : float, optional
        The model's learning rate
    hsqc_transformer : HsqcTransformer, optional
        The transformer for encoding HSQC
    spec_transformer : SpectraTransformer, optional
        The transformer for encoding Mass Spectra
    out_dim : int, optional
        The final output dimensionality of the model
    """
    def __init__(
            self,
            lr=1e-3,
            hsqc_transformer=HsqcTransformer(),
            spec_transformer=SpectraTransformer(),
            out_dim=6144
        ):

        self.lr = lr
        self.hsqc = hsqc_transformer
        self.spec = spec_transformer 
        self.fc = nn.Linear(self.hsqc.dim_model+self.spec.dim_model, out_dim)
        self.loss = nn.BCELoss()
        self.cos = nn.CosineSimilarity(dim=1)
    
    def forward(self, hsqc, ms):
        hsqc_encodings = self.hsqc.encode(hsqc)
        spec_encodings = self.spec.encode(ms)
        hsqc_cls_encoding = hsqc_encodings[:,:1,:].squeeze(1)
        spec_cls_encoding = spec_encodings[:,:1,:].squeeze(1)
        # TODO: make sure cat dimensionality is correct, not in batch dim
        print(spec_cls_encoding.shape)
        print(hsqc_cls_encoding.shape)
        out = torch.cat([hsqc_cls_encoding, spec_cls_encoding])
        print(out.shape)
        out = F.sigmoid(self.fc(out))
        return out

    def training_step(self, batch, batch_idx):
        hsqc, ms, fp = batch
        out = self.forward(hsqc, ms)
        loss = self.loss(out, fp)
        self.log("tr/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        hsqc, ms, fp = batch
        out = self.forward(hsqc, ms)
        loss = self.loss(out, fp)
        self.log("val/loss", loss)

        predicted = (out >= 0.5)
        # cosine similarity
        cos = self.cos(fp, predicted)
        self.log("val/cosine_sim", torch.mean(cos).item())

        # f1 score
        predicted = predicted.cpu()
        labels = fp.cpu()
        f1 = f1_score(predicted.flatten(), labels.flatten())
        self.log("val/f1", np.mean(f1))
        return loss
        
    def test_step(self, batch, batch_idx):
        hsqc, ms, fp = batch
        out = self.forward(hsqc, ms)
        loss = self.loss(out, fp)
        self.log("test/loss", loss)

        predicted = (out >= 0.5)
        # cosine similarity
        cos = self.cos(fp, predicted)
        self.log("test/cosine_sim", torch.mean(cos).item())

        # f1 score
        predicted = predicted.cpu()
        labels = fp.cpu()
        f1 = f1_score(predicted.flatten(), labels.flatten())
        self.log("test/f1", np.mean(f1))
        return loss
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)