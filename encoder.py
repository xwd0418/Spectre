import pytorch_lightning as pl
import torch, torch.nn as nn
from depthcharge.components import SpectrumEncoder

class MassSpecEncoder(pl.LightningModule):
    """
    A Transformer model to learn latent space embeddings of 
    mass spectra

    Parameters
    ----------
    dim_model : int, optional
        The latent dimensionality used by the Transformer model.
    n_head : int, optional
        The number of attention heads in each layer. ``dim_model`` must be
        divisible by ``n_head``.
    n_layers : int, optional
        The number of Transformer layers.
    dim_intensity: int or None, optional
        The number of features to use for encoding peak intensity.
        The remaining (``dim_model - dim_intensity``) are reserved for
        encoding the m/z value.
    dim_ff: int, optional
        The dimensionality of the fully connected layers in the Transformer
        Decoder layer of the model.
    dropout: float, optional
        The dropout probability for all layers.A
    """
    def __init__(
            self, 
            dim_model=128, 
            dim_intensity=32,
            n_head=4, 
            n_layers=4, 
            dim_ff=512,
            dropout=0,
            lr = 1e-3
        ):
        super().__init__()
        self.lr = lr
        self.loss = nn.BCELoss()

        self.enc = SpectrumEncoder(
            n_head=n_head, 
            dim_model=dim_model, 
            dim_intensity=dim_intensity, 
            n_layers=n_layers, 
            dropout=dropout
       )
        
        dec_layer = nn.TransformerDecoderLayer(
            d_model=dim_model,
            nhead=n_head,
            dim_feedforward=dim_ff,
            batch_first=True,
            dropout=dropout,
        )
        
        self.dec = nn.TransformerDecoder(
            dec_layer,
            num_layers=n_layers,
        )

        self.classify_token = nn.Embedding(1, dim_model)

        
    def forward(self, x):
        """
        Forward pass

        Parameters
        ----------
        spectra : torch.Tensor of shape (n_spectra, n_peaks, 2)
            The spectra to embed. Axis 0 represents a mass spectrum, axis 1
            contains the peaks in the mass spectrum, and axis 2 is essentially
            a 2-tuple specifying the m/z-intensity pair for each peak. These
            should be zero-padded, such that all of the spectra in the batch
            are the same length.

        """
        # TODO: implement precursor into model
        embeddings, _ = self.enc(x)
        # generate initial embedding token for decoder step
        tokens = self.classify_token(torch.zeros(x.size()[0], dtype=torch.long, device=self.device)).unsqueeze(1)
        output = self.dec(memory=embeddings, tgt=tokens)
        # output = self.fc(output)
        return output

    def training_step(self, batch, batch_idx):
        x, labels = batch

        # TODO: implement precursor into model
        embeddings, _ = self.enc(x)
        # generate initial embedding token for decoder step
        tokens = self.classify_token(torch.zeros(x.size()[0], dtype=torch.long, device=self.device)).unsqueeze(1)
        output = self.dec(memory=embeddings, tgt=tokens)
        output = self.fc(output)

        loss = self.loss(output, labels)
        self.log("tr/train", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, labels = batch

        # TODO: implement precursor into model
        embeddings, _ = self.enc(x)
        # generate initial embedding token for decoder step
        tokens = self.classify_token(torch.zeros(x.size()[0], dtype=torch.long, device=self.device)).unsqueeze(1)
        output = self.dec(memory=embeddings, tgt=tokens)
        output = self.fc(output)

        loss = self.loss(output, labels)
        self.log("val/loss", loss)
        return loss

    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = self.lr)
