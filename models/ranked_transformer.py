import logging
import pytorch_lightning as pl
import torch, torch.nn as nn
from encoder import CoordinateEncoder
from utils import ranker, constants
from models import compute_metrics
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import numpy as np, os

from utils.lr_scheduler import NoamOpt

class HsqcRankedTransformer(pl.LightningModule):
    """A Transformer encoder for input HSQC.
    Parameters
    ----------
    lr : float, optional
        The model's learning rate
    dim_model : int, optional
        The latent dimensionality to represent points on the HSQC
    dim_coords : tuple, optional
        A tuple (x,y,z) where x, y, and z are the number of dimensions to represent the
        each dimension of the hsqc coordinates. Must sum to dim_model
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
            dim_coords="43,43,42",
            heads=8,
            layers=1,
            ff_dim=1024,
            wavelength_bounds=None,
            dropout=0,
            out_dim=6144,
            save_params=True,
            module_only=False,
            pos_weight=1.0,
            weight_decay=0.0,
            scheduler=None, # None, "attention"
            **kwargs,
        ):
        super().__init__()
        params = locals().copy()
        self.out_logger = logging.getLogger("lightning")

        if save_params:
            self.save_hyperparameters(ignore=["save_params", "module_only"])

        if not module_only:
            self.ranker = ranker.RankingSet(file_path="./tempdata/hyun_pair_ranking_set_07_22/val_pair.pt")
            assert(os.path.exists("./tempdata/hyun_pair_ranking_set_07_22/val_pair.pt"))
            for k,v in params.items():
                if k not in constants.MODEL_LOGGING_IGNORE:
                    self.out_logger.info(f"Hparam: [{k}], value: [{v}]")
        
        dim_coords = tuple([int(v) for v in dim_coords.split(",")])
        assert(sum(dim_coords)==dim_model)

        self.lr = lr
        self.weight_decay = weight_decay

        self.enc = CoordinateEncoder(dim_model, dim_coords, wavelength_bounds)
        self.fc = nn.Linear(dim_model, out_dim)
        self.latent = torch.nn.Parameter(torch.randn(1, 1, dim_model))
        self.scheduler = scheduler
        self.dim_model = dim_model

        # The Transformer layers:
        layer = torch.nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=heads,
            dim_feedforward=ff_dim,
            batch_first=True,
            dropout=dropout,
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(
            layer,
            num_layers=layers,
        )
        self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.ones(out_dim) * pos_weight)

        self.out_logger.info("[RankedTransformer] Initialized")
    
    @staticmethod
    def add_model_specific_args(parent_parser, model_name=""):
        model_name = model_name if len(model_name) == 0 else f"{model_name}_"
        parser = parent_parser.add_argument_group(model_name)
        parser.add_argument(f"--{model_name}lr", type=float, default=1e-3)
        parser.add_argument(f"--{model_name}dim_model", type=int, default=128)
        parser.add_argument(f"--{model_name}dim_coords", type=str, default="43,43,42")
        parser.add_argument(f"--{model_name}heads", type=int, default=8)
        parser.add_argument(f"--{model_name}layers", type=int, default=8)
        parser.add_argument(f"--{model_name}ff_dim", type=int, default=1024)
        parser.add_argument(f"--{model_name}wavelength_bounds", type=int, default=None)
        parser.add_argument(f"--{model_name}dropout", type=float, default=0)
        parser.add_argument(f"--{model_name}out_dim", type=int, default=6144)
        parser.add_argument(f"--{model_name}pos_weight", type=float, default=1.0)
        parser.add_argument(f"--{model_name}weight_decay", type=float, default=0.0)
        parser.add_argument(f"--{model_name}scheduler", type=str, default=None)
        return parent_parser
    
    @staticmethod
    def prune_args(vals: dict, model_name=""):
        items = [(k[len(model_name)+1:],v) for k,v in vals.items() if k.startswith(model_name)]
        return dict(items)

    def encode(self, hsqc):
        """
        Returns
        -------
        latent : torch.Tensor of shape (n_spectra, n_peaks + 1, dim_model)
            The latent representations for the spectrum and each of its
            peaks.
        mem_mask : torch.Tensor
            The memory mask specifying which elements were padding in X.
        """
        zeros = ~hsqc.sum(dim=2).bool()
        mask = [
            torch.tensor([[False]] * hsqc.shape[0]).type_as(zeros),
            zeros,
        ]
        mask = torch.cat(mask, dim=1)
        points = self.enc(hsqc)

        # Add the spectrum representation to each input:
        latent = self.latent.expand(points.shape[0], -1, -1)

        points = torch.cat([latent, points], dim=1)
        out = self.transformer_encoder(points, src_key_padding_mask=mask)
        return out

    def forward(self, hsqc):
        """The forward pass.
        Parameters
        ----------
        hsqc: torch.Tensor of shape (batch_size, n_points, 3)
            The hsqc to embed. Axis 0 represents an hsqc, axis 1
            contains the coordinates in the hsqc, and axis 2 is essentially is
            a 3-tuple specifying the coordinate's x, y, and z value. These
            should be zero-padded, such that all of the hsqc in the batch
            are the same length.
        """
        out = self.encode(hsqc)
        out = self.fc(out[:,:1,:].squeeze(1)) # extracts cls token
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
        return compute_metrics.cm(out, labels, self.ranker, loss, thresh = 0.0)


    def validation_epoch_end(self, validation_step_outputs):
        feats = validation_step_outputs[0].keys()
        di = {}
        for feat in feats:
            di[f"val/mean_{feat}"] = np.mean([v[feat] for v in validation_step_outputs])
        for k,v in di.items():
            self.log(k, v, on_epoch=True)

    def configure_optimizers(self):
        if not self.scheduler:
            return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.scheduler == "attention":
            optim = torch.optim.Adam(self.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9, weight_decay=self.weight_decay)
            scheduler = NoamOpt(self.dim_model, 4000, optim)
            return {
                "optimizer": optim,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                }
            }