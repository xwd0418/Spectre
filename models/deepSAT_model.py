import logging
import pytorch_lightning as pl
import torch, pickle
import math
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import torch.distributed as dist
from datasets.dataset_utils import specific_radius_mfp_loader
from models import compute_metrics

from utils import ranker, constants
from models import compute_metrics

import numpy as np
import os


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.bn0 = nn.BatchNorm2d(in_channels)
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, (3, 3), padding='same')
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, (3, 3), padding='same')
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels, (3, 3), padding='same')
        self.bn3 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x_shortcut = x
        x = F.relu(self.bn0(x))
        
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = torch.cat([x, x_shortcut], dim=1)
        return x

class DeepSATModel(pl.LightningModule):
    def __init__(self, 
                 num_classes, 
                 ranking_set_path="",
                 *args,
                **kwargs,
                ):
        super(DeepSATModel, self).__init__()
        self.conv1 = nn.Conv2d(2, 64, (3, 3), padding="same")
        self.conv2 = nn.Conv2d(64, 64, (3, 3), padding="same")
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d((2, 2))
        self.resblock1 = ResidualBlock(64,128)
        self.resblock2 = ResidualBlock(128+64,256) # 128+64=192
        self.resblock3 = ResidualBlock(256+192,512) # 256+192=448
        self.resblock4 = ResidualBlock(512+448,1024) # 512+448=960
        self.resblock5 = ResidualBlock(1024+960,2048) # 1024+960=1984
        self.global_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fp_output = nn.Linear(2048+1984, 6144)
        self.mw_output = nn.Linear(2048+1984, 1)
        self.class_output = nn.Linear(2048+1984, num_classes)
        self.glycoside_output = nn.Linear(2048+1984, 2)

        self.BCE_loss = nn.BCEWithLogitsLoss()
        self.CE_loss  = nn.CrossEntropyLoss(reduction='sum')
        
        # additional variables
        self.bs = kwargs['bs']
        self.lr = kwargs['lr']
        self.loss_weight_ratio = kwargs['loss_weight_ratio']
        
        self.rank_by_soft_output = kwargs['rank_by_soft_output']
        self.validation_step_outputs = []
        self.training_step_outputs = []
        self.test_step_outputs = []
        
        # ranking/retrieval
        self.compute_metric_func = compute_metrics.cm
        self.use_actaul_mw_for_retrival =  kwargs['use_actaul_mw_for_retrival']
        
        self.ranking_set_path = '/root/MorganFP_prediction/reproduce_previous_works/reproducing_deepsat/ranking_sets/HYUN_FP_only_all_info_molecules/val/rankingset.pt'
        # print(ranking_set_path)
        assert os.path.exists(self.ranking_set_path), f"{ranking_set_path} does not exist"
        self.ranker = ranker.RankingSet(file_path=self.ranking_set_path, batch_size=self.bs, use_actaul_mw_for_retrival=self.use_actaul_mw_for_retrival)

        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn1(self.conv2(x)))
        x = self.pool(x)
        x = self.pool(self.resblock1(x))
        x = self.pool(self.resblock2(x))
        x = self.pool(self.resblock3(x))
        x = self.pool(self.resblock4(x))
        x = self.global_pool(self.resblock5(x))
        # x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(x)
        fp = self.fp_output(x)
        mw = self.mw_output(x)
        np_cls = self.class_output(x)
        gly = self.glycoside_output(x)
        return fp, mw, np_cls, gly

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        images, targets = batch
        fp, mw, np_cls, gly = self(images)
        loss_fp = self.BCE_loss(fp, targets["fp"])
        
        loss_mw = F.mse_loss(mw, targets["mw"]) / self.loss_weight_ratio
        
        # ingore loss where targets are -1
        sum_non_ignore_cls = torch.sum(targets["np_cls"]!=-1)
        np_cls[targets["np_cls"]==-1] = 0
        targets["np_cls"][targets["np_cls"]==-1] = 0
        
        sum_non_ignore_gly = torch.sum(targets["gly"]!=-1)
        gly[targets["gly"]==-1] = 0
        targets["gly"][targets["gly"]==-1] = 0
        # #
        loss_np_cls = self.CE_loss(np_cls, targets["np_cls"]) / sum_non_ignore_cls / self.loss_weight_ratio
        loss_gly = self.CE_loss(gly, targets["gly"]) / sum_non_ignore_gly / self.loss_weight_ratio
       
        total_loss = loss_fp + loss_mw + loss_np_cls + loss_gly
        self.log('train_loss', total_loss, prog_bar=True)
        return total_loss
    
    
    def validation_step(self, batch, batch_idx):
     
        images, targets = batch
        fp, mw, np_cls, gly = self(images)
        
        preds = fp
        # print("preds min and max is ", preds.min(), preds.max())
        # print((labels) )
        # print("\n\n\n")
        loss_fp = self.BCE_loss(fp, targets["fp"])
        mw_info_for_retrieval = targets["mw"] if self.use_actaul_mw_for_retrival else mw
        metrics = self.compute_metric_func(
            preds, targets["fp"], self.ranker, loss_fp, self.BCE_loss, thresh=0.0, 
            rank_by_soft_output=self.rank_by_soft_output,
            query_idx_in_rankingset=batch_idx,
            # mw=mw_info_for_retrieval,
            # use_actaul_mw_for_retrival=self.use_actaul_mw_for_retrival
            )
        if type(self.validation_step_outputs)==list: # adapt for child class: optional_input_ranked_transformer
            self.validation_step_outputs.append(metrics)
        return metrics
    
    def test_step(self, batch, batch_idx):
        images, targets = batch
        fp, mw, np_cls, gly = self(images)
        
        preds = fp
        # print((labels) )
        # print("\n\n\n")
        loss_fp = self.BCE_loss(fp, targets["fp"])
        mw_info_for_retrieval = targets["mw"] if self.use_actaul_mw_for_retrival else mw
        metrics = self.compute_metric_func(
            preds, targets["fp"], self.ranker, loss_fp, self.BCE_loss, thresh=0.0, 
            rank_by_soft_output=self.rank_by_soft_output,
            query_idx_in_rankingset=batch_idx,
            # mw=mw_info_for_retrieval,
            # use_actaul_mw_for_retrival=self.use_actaul_mw_for_retrival
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
        
        
    def change_ranker_for_testing(self, test_ranking_set_path=None ):
       
        if test_ranking_set_path is None:
            test_ranking_set_path = self.ranking_set_path.replace("val", "test")
        self.ranker = ranker.RankingSet(file_path=test_ranking_set_path, batch_size=self.bs, use_actaul_mw_for_retrival=self.use_actaul_mw_for_retrival)
        
        
    @staticmethod
    def add_model_specific_args(parent_parser, model_name=""):
        model_name = model_name if len(model_name) == 0 else f"{model_name}_"
        parser = parent_parser.add_argument_group(model_name)
        parser.add_argument(f"--{model_name}use_actaul_mw_for_retrival", type=lambda x:bool(str2bool(x)) )
        parser.add_argument(f"--{model_name}lr", type=float, default=0.00001)
        # loss_weight_ratio
        parser.add_argument(f"--{model_name}loss_weight_ratio", type=float, default=10000)
        
        return parent_parser

def str2bool(v):    
        # specifically used for arg-paser with boolean values
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.') 
