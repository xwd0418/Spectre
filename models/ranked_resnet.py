import logging
import pytorch_lightning as pl
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import torch.distributed as dist
import torchvision

from utils import ranker, constants
from models import compute_metrics
from models.encoders.encoder_factory import build_encoder
from models.extras.transformer_stuff import (
    generate_square_subsequent_mask
)
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import numpy as np
import os


# from models.ranked_transformer import HsqcRankedResNet


class HsqcRankedResNet(pl.LightningModule):
    """A Transformer encoder for input HSQC.
    Parameters
    ----------
    """

    def __init__(
        self,
        # model args

        dropout=0,
        out_dim=6144,
        # other business logic stuff
        save_params=True,
        ranking_set_path="",
        FP_choice="R2-6144FP",
        loss_func = "",
        # training args
        lr=1e-5,
        pos_weight = None,
        weight_decay = 0.0,
        num_input_channels=0,
        scheduler=None,   # cosine annealing
        freeze_weights=False,
        *args,
        **kwargs,
    ):
        super().__init__()
        params = locals().copy()
        self.out_logger = logging.getLogger("lightning")
        if dist.is_initialized():
            rank = dist.get_rank()
            if rank != 0:
                # For any process with rank other than 0, set logger level to WARNING or higher
                self.out_logger.setLevel(logging.WARNING)
        self.out_logger.info("[RankedResNet] Started Initializing")
        self.logger_should_sync_dist = torch.cuda.device_count() > 1


        # === All Parameters ===
        self.FP_length = out_dim # 6144 
        if FP_choice == "R0_to_R4_30720_FP":
            out_dim = self.FP_length * 5
        if loss_func == "CE":
            assert(FP_choice == "R2-6144-count-based-FP")
            out_dim = self.FP_length * kwargs['num_class']
        self.bs = kwargs['bs']
        self.num_class = kwargs['num_class'] if loss_func == "CE" else None 
        self.lr = lr
        self.weight_decay = weight_decay

        self.scheduler = scheduler
  
        
        # don't set ranking set if you just want to treat it as a module
        if ranking_set_path:
            self.ranking_set_path = ranking_set_path
            # print(ranking_set_path)
            assert (os.path.exists(ranking_set_path))
            self.rank_by_soft_output = kwargs['rank_by_soft_output']
            self.ranker = ranker.RankingSet(file_path=ranking_set_path, batch_size=self.bs, CE_num_class=self.num_class)

        if save_params:
            print("HsqcRankedResNet saving args")
            self.save_hyperparameters(*kwargs.keys())


        ### Loss function 
        if pos_weight==None:
            self.bce_pos_weight = None
            self.out_logger.info("[RankedResNet] bce_pos_weight = None")
        else:
            try:
                pos_weight_value = float(pos_weight)
                self.bce_pos_weight= torch.full((self.FP_length,), pos_weight_value)
                self.out_logger.info(f"[RankedResNet] bce_pos_weight is {pos_weight_value}")
        
            except ValueError:
                if pos_weight == "ratio":
                    self.bce_pos_weight = torch.load('/root/MorganFP_prediction/reproduce_previous_works/smart4.5/pos_weight_array_based_on_ratio.pt')
                    self.out_logger.info("[RankedResNet] bce_pos_weight is loaded ")
                else:
                    raise ValueError(f"pos_weight {pos_weight} is not valid")
        
        self.loss_func = loss_func
        if FP_choice == "R2-6144-count-based-FP":
            if loss_func == "MSE":
                self.loss = nn.MSELoss()
                self.compute_metric_func = compute_metrics.cm_count_based_mse
            elif loss_func == "CE":
                self.loss = nn.CrossEntropyLoss()
                self.compute_metric_func = compute_metrics.cm_count_based_ce
            else:
                raise Exception("loss_func should be either MSE or CE when using count-based FP")
        else:
            self.loss = nn.BCEWithLogitsLoss(pos_weight=self.bce_pos_weight)
            self.compute_metric_func = compute_metrics.cm
        
        
        
        # additional nn modules 
        self.validation_step_outputs = []
        self.training_step_outputs = []
        self.test_step_outputs = []

        self.resnet = torchvision.models.resnet50()
            
        # configure first convolutional layer to based on num channels
        self.resnet.conv1 = nn.Conv2d(num_input_channels, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)    
        
            
        in_feat = self.resnet.fc.in_features
        print(f"resnet fc layer in_features size: {in_feat}")
        self.resnet.fc = nn.Linear(in_feat, out_dim)
        self.dropout = nn.Dropout(dropout)
        # (1, 1, dim_model)

        if freeze_weights:
            self.out_logger.info("[RankedResNet] Freezing Weights")
            for parameter in self.parameters():
                parameter.requires_grad = False
        self.out_logger.info("[RankedResNet] Initialized")

    @staticmethod
    def add_model_specific_args(parent_parser, model_name=""):
        model_name = model_name if len(model_name) == 0 else f"{model_name}_"
        parser = parent_parser.add_argument_group(model_name)
        parser.add_argument(f"--{model_name}lr", type=float, default=5e-5)
       
        parser.add_argument(f"--{model_name}dropout", type=float, default=0.2)
        parser.add_argument(f"--{model_name}num_input_channels", type=int, default=2)
        # parser.add_argument(f"--{model_name}out_dim", type=int, default=6144)
        parser.add_argument(f"--{model_name}pos_weight", type=str, default=None, 
                            help = "if none, then not to be used; if ratio,\
                                then used the save tensor which is the ratio of num_0/num_1, \
                                if float num ,then use this as the ratio")
        parser.add_argument(f"--{model_name}weight_decay", type=float, default=1e-6)
      
        parser.add_argument(f"--{model_name}scheduler", type=str, default=None)
        parser.add_argument(f"--{model_name}freeze_weights",
                            type=bool, default=False)
        parser.add_argument(
            f"--{model_name}ranking_set_path", type=str, default="")
        return parent_parser

    @staticmethod
    def prune_args(vals: dict, model_name=""):
        items = [(k[len(model_name) + 1:], v)
                 for k, v in vals.items() if k.startswith(model_name)]
        return dict(items)


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
        out = self.resnet(hsqc) 
        out = self.dropout(out) 
        return out

    def training_step(self, batch, batch_idx):
        x, labels = batch
        out = self.forward(x)
        if self.loss_func == "CE":
            out = out.view(out.shape[0],  self.num_class, self.FP_length)
        loss = self.loss(out, labels)

        self.log("tr/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        
        x, labels = batch
        out = self.forward(x)
        if self.loss_func == "CE":
            out = out.view(out.shape[0],  self.num_class, self.FP_length)
            preds = out.argmax(dim=1)
        else:
            preds = out
        loss = self.loss(out, labels)
        metrics = self.compute_metric_func(
            preds, labels, self.ranker, loss, self.loss, thresh=0.0, 
            rank_by_soft_output=self.rank_by_soft_output,
            query_idx_in_rankingset=batch_idx
            )
        if type(self.validation_step_outputs)==list: # adapt for child class: optional_input_ranked_transformer
            self.validation_step_outputs.append(metrics)
        return metrics
    
    def test_step(self, batch, batch_idx):
        x, labels = batch
        out = self.forward(x)
        if self.loss_func == "CE":
            out = out.view(out.shape[0],  self.num_class, self.FP_length)
            preds = out.argmax(dim=1)
        else:
            preds = out
        loss = self.loss(out, labels)
        metrics = self.compute_metric_func(
            preds, labels, self.ranker, loss, self.loss, thresh=0.0,
            rank_by_soft_output=self.rank_by_soft_output,
            query_idx_in_rankingset=batch_idx
            )
        if type(self.test_step_outputs)==list:
            self.test_step_outputs.append(metrics)
        return metrics

   
            
    def on_train_epoch_end(self):
        # return
        if self.training_step_outputs:
            feats = self.training_step_outputs[0].keys()
            di = {}
            for feat in feats:
                di[f"tr/mean_{feat}"] = np.mean([v[feat]
                                                for v in self.training_step_outputs])
            for k, v in di.items():
                self.log(k, v, on_epoch=True)
            self.training_step_outputs.clear()

    def on_validation_epoch_end(self):
        # return
        feats = self.validation_step_outputs[0].keys()
        di = {}
        for feat in feats:
            di[f"val/mean_{feat}"] = np.mean([v[feat]
                                             for v in self.validation_step_outputs])
        for k, v in di.items():
            self.log(k, v, on_epoch=True, prog_bar=k=="val/mean_rank_1")
        self.validation_step_outputs.clear()
        
    def on_test_epoch_end(self):
        feats = self.test_step_outputs[0].keys()
        di = {}
        for feat in feats:
            di[f"test/mean_{feat}"] = np.mean([v[feat]
                                             for v in self.test_step_outputs])
        for k, v in di.items():
            self.log(k, v, on_epoch=True, sync_dist=False)
            # self.log(k, v, on_epoch=True)
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                    lr=self.lr,
                                    # momentum=0.9, 
                                    weight_decay=self.weight_decay)
                
        if self.scheduler is None:
            return optimizer            
        
        if self.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1700)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                }
            }

    def log(self, name, value, *args, **kwargs):
        # Set 'sync_dist' to True by default
        if kwargs.get('sync_dist') is None:
            kwargs['sync_dist'] = kwargs.get(
                'sync_dist', self.logger_should_sync_dist)
        if name == "test/mean_rank_1":
            print(kwargs,"\n\n")
        super().log(name, value, *args, **kwargs)
        
    def change_ranker_for_testing(self):
        test_ranking_set_path = self.ranking_set_path.replace("val", "test")
        self.ranker = ranker.RankingSet(file_path=test_ranking_set_path, batch_size=self.bs,  CE_num_class=self.num_class)

