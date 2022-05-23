import pytorch_lightning as pl
import torch, torch.nn as nn 
from models.spectra_transformer import SpectraTransformer 
from models.hsqc_transformer import HsqcTransformer
from sklearn.metrics import f1_score  
import numpy as np

class BaselineDoubleTransformer(pl.LightningModule):
    """ A baseline implementation of both hsqc and ms transformers
    Parameters
    ----------
    lr : float, optional
        The model's learning rate
    hsqc_transformer : HsqcTransformer, optional
        The transformer for encoding HSQC
    spec_transformer : SpectraTransformer, optional
        The transformer for encoding Mass Spectra
    num_hidden : int, optional
        The number of hidden layers to use
    fc_dim : int, optional
        The dimensionality of the feedforward hidden layers
    out_dim : int, optional
        The final output dimensionality of the model
    """
    def __init__(
            self,
            lr=1e-3,
            hsqc_transformer=HsqcTransformer(),
            spec_transformer=SpectraTransformer(),
            num_hidden=0, 
            fc_dim=128, 
            dropout=0,
            out_dim=6144
        ):

        super().__init__()
        self.lr = lr
        self.hsqc = hsqc_transformer
        self.spec = spec_transformer 
        total_emb_dim = self.hsqc.fc.in_features+self.spec.fc.in_features

        if num_hidden > 0:
            fc_layers = [nn.Linear(total_emb_dim, fc_dim), nn.ReLU(), nn.Dropout(dropout)]
            for i in range(num_hidden):
                fc_layers.append(nn.Linear(fc_dim, fc_dim))
                fc_layers.append(nn.ReLU())
                fc_layers.append(nn.Dropout(dropout))
            fc_layers.append(nn.Linear(fc_dim, out_dim))
            fc_layers.append(nn.Sigmoid())
            self.fc = nn.Sequential(*fc_layers)
        else:
            self.fc = nn.Sequential(
                nn.Linear(total_emb_dim, out_dim),
                nn.sigmoid()
            )

        self.loss = nn.BCELoss()
        self.cos = nn.CosineSimilarity(dim=1)
    
    def forward(self, hsqc, ms):
        hsqc_encodings = self.hsqc.encode(hsqc)
        spec_encodings = self.spec.encode(ms)
        hsqc_cls_encoding = hsqc_encodings[:,:1,:].squeeze(1)
        spec_cls_encoding = spec_encodings[:,:1,:].squeeze(1)
        out = torch.cat([hsqc_cls_encoding, spec_cls_encoding], dim=1)
        out = self.fc(out)
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
        return {"test_loss":loss, "test_f1":f1}
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)