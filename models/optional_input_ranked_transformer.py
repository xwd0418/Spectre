from models.ranked_transformer import HsqcRankedTransformer
from collections import defaultdict
import numpy as np
import torch.nn as nn

class OptionalInputRankedTransformer(HsqcRankedTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.validation_step_outputs = defaultdict(list)
        self.test_step_outputs = defaultdict(list)
        self.all_dataset_names = ["all_inputs", "HSQC_H_NMR", "HSQC_C_NMR", "only_hsqc", "only_1d", "only_H_NMR",  "only_C_NMR"]
        if self.separate_classifier:
            self.classifiers = nn.ModuleList([nn.Linear(self.dim_model, self.out_dim) for i in range(len(self.all_dataset_names))])
            self.fc = None
    
    def training_step(self, batch, batch_idx):
        if not self.separate_classifier:
            return super().training_step(batch, batch_idx)
        
        x, labels, input_type = batch
        out, _ = self.encode(x)
        cls_token = out[:, :1, :].squeeze(1)
        total_loss = 0
        for i, classifier in enumerate(self.classifiers):
            cls_for_i = cls_token[input_type == i]
            if len(cls_for_i) == 0:
                continue
            out = classifier(cls_for_i)
            loss = self.loss(out, labels[input_type == i])
            total_loss += loss
            self.log(f"tr/loss_{self.all_dataset_names[i]}", loss.item()/len(cls_for_i))
        # loss = self.loss(out, labels)

        # self.log("tr/loss", loss, prog_bar=True)
        return total_loss
            
            
    def validation_step(self, batch, batch_idx, dataloader_idx ):
        if self.separate_classifier:
            self.fc = self.classifiers[dataloader_idx]
        current_batch_name = self.all_dataset_names[dataloader_idx]
        metrics = super().validation_step(batch, batch_idx)
        self.validation_step_outputs[current_batch_name].append(metrics)
        return metrics
    
    def test_step(self, batch, batch_idx, dataloader_idx):
        if self.separate_classifier:
            self.fc = self.classifiers[dataloader_idx]
        current_batch_name = self.all_dataset_names[dataloader_idx]
        metrics = super().test_step(batch, batch_idx)
        self.test_step_outputs[current_batch_name].append(metrics)
        return metrics
    
    def on_validation_epoch_end(self):
        # return
        for dataset_name in self.all_dataset_names:
            feats = self.validation_step_outputs[dataset_name][0].keys()
            di = {}
            for feat in feats:
                di[f"val_mean_{feat}_{dataset_name}"] = np.mean([v[feat]
                                                 for v in self.validation_step_outputs[dataset_name]])
            for k, v in di.items():
                self.log(k, v, on_epoch=True, prog_bar="rank_1" in k)
        self.validation_step_outputs.clear()
        
        
    def on_test_epoch_end(self):
        for dataset_name in self.all_dataset_names:
            feats = self.test_step_outputs[dataset_name][0].keys()
            di = {}
            for feat in feats:
                di[f"test_mean_{feat}_{dataset_name}"] = np.mean([v[feat]
                                                 for v in self.test_step_outputs[dataset_name]])
            for k, v in di.items():
                self.log(k, v, on_epoch=True)
        self.test_step_outputs.clear()