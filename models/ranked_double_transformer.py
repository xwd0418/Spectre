import pytorch_lightning as pl
import torch, torch.nn as nn
from models.ranked_transformer import HsqcRankedTransformer 
import numpy as np, os, logging
from utils import ranker

class DoubleTransformer(pl.LightningModule):
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
    dropout : float, optiona
        The dropout probability
    out_dim : int, optional
        The final output dimensionality of the model
    """
    def __init__(
            self,
            lr=1e-3,
            num_hidden=0, 
            fc_dim=128, 
            dropout=0,
            out_dim=6144,
            pos_weight=1.0,
            ms_dim_model=128,
            ms_dim_coords="43,43,42",
            ms_heads=8,
            ms_layers=1,
            ms_ff_dim=1024,
            ms_wavelength_bounds=None,
            ms_dropout=0,
            ms_out_dim=6144,
            hsqc_dim_model=128,
            hsqc_dim_coords="43,43,42",
            hsqc_heads=8,
            hsqc_layers=1,
            hsqc_ff_dim=1024,
            hsqc_wavelength_bounds=None,
            hsqc_dropout=0,
            hsqc_out_dim=6144,
            hsqc_weights=None,
            ms_weights=None,
            ranking_sets={
                "editfp_val": "./tempdata/hyun_pair_ranking_set_07_22/val_pair.pt",
            },
            **kwargs
        ):
        super().__init__()
        params = locals().copy()
        self.save_hyperparameters(ignore=["ranking_sets"])

        self.lr = lr
        hsqc = HsqcRankedTransformer.prune_args(params, "hsqc")
        ms = HsqcRankedTransformer.prune_args(params, "ms")
        if hsqc_weights is not None:
            self.hsqc = HsqcRankedTransformer.load_from_checkpoint(hsqc_weights)
            print(f"[DoubleTransformer] Loading HSQC Model: {hsqc_weights}")
        else:
            self.hsqc = HsqcRankedTransformer(save_params=False, module_only=True, **hsqc)

        if ms_weights:
            self.spec = HsqcRankedTransformer.load_from_checkpoint(ms_weights)
            print(f"[DoubleTransformer] Loading MS Model: {ms_weights}")
        else:
            self.spec = HsqcRankedTransformer(save_params=False, module_only=True, **ms)
        
        total_emb_dim = self.hsqc.fc.in_features+self.spec.fc.in_features

        if num_hidden > 0:
            fc_layers = [nn.Linear(total_emb_dim, fc_dim), nn.ReLU(), nn.Dropout(dropout)]
            for _ in range(num_hidden):
                fc_layers.append(nn.Linear(fc_dim, fc_dim))
                fc_layers.append(nn.ReLU())
                fc_layers.append(nn.Dropout(dropout))
            fc_layers.append(nn.Linear(fc_dim, out_dim))
            self.fc = nn.Sequential(*fc_layers)
        else:
            self.fc = nn.Sequential(
                nn.Linear(total_emb_dim, out_dim),
            )

        self.rankers = {}
        for k,v in ranking_sets.items():
            assert(os.path.exists(v))
            self.rankers[k] = ranker.RankingSet(file_path=v)

        self.loss = self.loss = nn.BCEWithLogitsLoss()
    
    @staticmethod
    def add_model_specific_args(parent_parser, model_name="tnsfm"):
        parser = parent_parser.add_argument_group(model_name)
        parser.add_argument(f"--lr", type=float, default=1e-3)
        parser.add_argument(f"--num_hidden", type=int, default=0)
        parser.add_argument(f"--fc_dim", type=int, default=128)
        parser.add_argument(f"--dropout", type=float, default=0)
        parser.add_argument(f"--out_dim", type=int, default=6144)
        parser.add_argument(f"--hsqc_weights", type=str, default=None)
        parser.add_argument(f"--ms_weights", type=str, default=None)
        parser.add_argument(f"--pos_weight", type=float, default=1.0)
        HsqcRankedTransformer.add_model_specific_args(parser, "hsqc")
        HsqcRankedTransformer.add_model_specific_args(parser, "ms")
        return parent_parser

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

        predicted = (out >= 0.5)
        # cosine similarity
        cos = self.cos(fp, predicted)
        # active bits
        active = torch.mean(torch.sum(predicted, axis=1) / 6144.0)
        # f1 score
        predicted = predicted.cpu()
        labels = fp.cpu()
        f1 = f1_score(labels.flatten(), predicted.flatten())
        precision = precision_score(labels.flatten(), predicted.flatten())
        recall = recall_score(labels.flatten(), predicted.flatten())
        accuracy = accuracy_score(labels.flatten(), predicted.flatten())

        # ranking f1
        predicted = predicted.type(torch.FloatTensor)
        ranks = {}
        for k,v in self.rankers.items():
            rank_res = v.batched_rank(predicted, labels)
            cts = [1, 5, 10]
            local_ranks = {f"rank_{allow}_{k}": torch.sum(rank_res < allow)/len(rank_res) for allow in cts}
            ranks.update(local_ranks)
        return {"ce_loss": loss.item(), "cos": torch.mean(cos).item(), 
            "f1": np.mean(f1), "precision": np.mean(precision), "recall": np.mean(recall), 
            "accuracy": np.mean(accuracy), "perc_active_bits": active.item(), **ranks}
    
    def validation_epoch_end(self, validation_step_outputs):
        feats = validation_step_outputs[0].keys()
        di = {}
        for feat in feats:
            di[f"val/mean_{feat}"] = np.mean([v[feat] for v in validation_step_outputs])
        for k,v in di.items():
            self.log(k, v)

    def test_step(self, batch, batch_idx):
        hsqc, ms, fp = batch
        out = self.forward(hsqc, ms)
        loss = self.loss(out, fp)

        predicted = (out >= 0.5)
        # cosine similarity
        cos = self.cos(fp, predicted)
        # active bits
        active = torch.mean(torch.sum(predicted, axis=1) / 6144.0)
        # f1 score
        predicted = predicted.cpu()
        labels = fp.cpu()
        f1 = f1_score(labels.flatten(), predicted.flatten())
        precision = precision_score(labels.flatten(), predicted.flatten())
        recall = precision_score(labels.flatten(), predicted.flatten())
        accuracy = accuracy_score(labels.flatten(), predicted.flatten())

        # ranking f1
        predicted = predicted.type(torch.FloatTensor)
        ranks = {}
        for k,v in self.rankers.items():
            rank_res = v.batched_rank(predicted, labels)
            cts = [1, 5, 10, 20]
            local_ranks = {f"rank_{allow}_{k}": torch.sum(rank_res < allow)/len(rank_res) for allow in cts}
            ranks.update(local_ranks)
        return {"ce_loss": loss.item(), "cos": torch.mean(cos).item(), 
            "f1": np.mean(f1), "precision": np.mean(precision), "recall": np.mean(recall), 
            "accuracy": np.mean(accuracy), "perc_active_bits": active.item(), **ranks}
    
    def test_step_end(self, test_step_outputs):
        feats = test_step_outputs[0].keys()
        di = {}
        for feat in feats:
            di[f"test/mean_{feat}"] = np.mean([v[feat] for v in test_step_outputs])
        for k,v in di.items():
            self.log(k, v)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
