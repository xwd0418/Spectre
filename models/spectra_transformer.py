import pytorch_lightning as pl
import torch, torch.nn as nn
from encoder import CoordinateEncoder
from sklearn.metrics import f1_score  
import numpy as np

class SpectraTransformer(pl.LightningModule):
    """A Transformer encoder for input mass spectra.
    Parameters
    ----------
    lr : float, optional
        The model's learning rate
    dim_model : int, optional
        The latent dimensionality to represent peaks in the mass spectrum.
    dim_coords : tuple, optional
        A tuple (x,y) where x and y are the number of dimensions to represent the
        m/z and intensity of the spectra, respectively. Must sum to dim_model
    n_head : int, optional
        The number of attention heads in each layer. ``dim_model`` must be
        divisible by ``n_head``.
    dim_feedforward : int, optional
        The dimensionality of the fully connected layers in the Transformer
        layers of the model.
    n_layers : int, optional
        The number of Transformer layers.
    wavelength_bounds : list(tuple), optional
        A list of tuples of (minimum, maximum) wavelengths for
        each dimension to be encoded 
    dropout : float, optional
        The dropout probability for all layers.
    out_dim : int, optional
        The final output dimensionality of the model
    """
    def __init__(
            self,
            lr=1e-3,
            dim_model=128,
            dim_coords=(64,64),
            n_heads=8,
            dim_feedforward=1024,
            n_layers=1,
            wavelength_bounds=None,
            dropout=0,
            out_dim=6144
        ):
        super().__init__()
        self.lr = lr
        self.enc = CoordinateEncoder(dim_model, dim_coords, wavelength_bounds)
        self.fc = nn.Linear(dim_model, out_dim)
        self.sigmoid = nn.Sigmoid()
        self.latent_spectrum = torch.nn.Parameter(torch.randn(1, 1, dim_model))

        # The Transformer layers:
        layer = torch.nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout,
        )

        self.transformer_encoder = torch.nn.TransformerEncoder(
            layer,
            num_layers=n_layers,
        )

        self.loss = nn.BCELoss()
        self.cos = nn.CosineSimilarity(dim=1)
    
    def encode(self, spectra):
        """
        Returns
        -------
        latent : torch.Tensor of shape (n_spectra, n_peaks + 1, dim_model)
            The latent representations for the spectrum and each of its
            peaks.
        mem_mask : torch.Tensor
            The memory mask specifying which elements were padding in X.
        """
        zeros = ~spectra.sum(dim=2).bool()
        mask = [
            torch.tensor([[False]] * spectra.shape[0]).type_as(zeros),
            zeros,
        ]
        mask = torch.cat(mask, dim=1)
        peaks = self.enc(spectra)

        # Add the spectrum representation to each input:
        latent_spectra = self.latent_spectrum.expand(peaks.shape[0], -1, -1)

        peaks = torch.cat([latent_spectra, peaks], dim=1)
        out = self.transformer_encoder(peaks, src_key_padding_mask=mask)
        return out

    def forward(self, spectra):
        """The forward pass.
        Parameters
        ----------
        spectra : torch.Tensor of shape (n_spectra, n_peaks, 2)
            The spectra to embed. Axis 0 represents a mass spectrum, axis 1
            contains the peaks in the mass spectrum, and axis 2 is essentially
            a 2-tuple specifying the m/z-intensity pair for each peak. These
            should be zero-padded, such that all of the spectra in the batch
            are the same length.
        """
        out = self.encode(spectra)
        out = torch.sigmoid(self.fc(out[:,:1,:].squeeze(1)))
        return out
    
    def training_step(self, batch, batch_idx):
        x, labels = batch
        out = self.forward(x)
        loss = self.loss(out, labels)
        self.log("tr/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, labels = batch
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
        x, labels = batch
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