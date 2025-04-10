from models.ranked_transformer import HsqcRankedTransformer
from collections import defaultdict
import numpy as np
import torch.nn as nn

class OptionalInputRankedTransformer(HsqcRankedTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.validation_step_outputs = defaultdict(list)
        self.test_step_outputs = defaultdict(list)
        self.test_np_classes_rank1 = defaultdict(lambda : defaultdict(list))
        self.all_dataset_names = ["all_inputs", "HSQC_H_NMR", "HSQC_C_NMR", "only_hsqc", "only_1d", "only_H_NMR",  "only_C_NMR"]
        if self.separate_classifier:
            self.classifiers = nn.ModuleList([nn.Linear(self.dim_model, self.out_dim) for i in range(len(self.all_dataset_names))])
            self.fc = None
        self.loader_idx = None    
        
    def only_test_this_loader(self, loader_idx):
        self.loader_idx = loader_idx
    
    def training_step(self, batch, batch_idx):
        if not self.separate_classifier:
            return super().training_step(batch, batch_idx)
        
        inputs, labels, NMR_type_indicator, input_type = batch
        out = self.forward(inputs, NMR_type_indicator)
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
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        if self.loader_idx != None  and self.loader_idx != dataloader_idx:
            # print(f"skipping loader {dataloader_idx} because self.loader_idx is {self.loader_idx}")
            return
        if self.separate_classifier:
            self.fc = self.classifiers[dataloader_idx]
        current_batch_name = self.all_dataset_names[dataloader_idx]
        metrics, np_classes, rank_1_hits = super().test_step(batch, batch_idx)
        self.test_step_outputs[current_batch_name].append(metrics)
        
        for curr_classes, curr_rank_1_hits in zip(np_classes, rank_1_hits.tolist()):
                for np_class in curr_classes:
                    self.test_np_classes_rank1[current_batch_name][np_class].append(curr_rank_1_hits)
        
        # return metrics
    
    def predict_step(self, batch, batch_idx, dataloader_idx,return_representations=False):
        if self.separate_classifier:
            self.fc = self.classifiers[dataloader_idx]
        current_batch_name = self.all_dataset_names[dataloader_idx]
        query_products = super().predict_step(batch, batch_idx, return_representations)
        return query_products
    
    def on_validation_epoch_end(self):
        # return
        total_features = defaultdict(list)
        for dataset_name in self.all_dataset_names:
            feats = self.validation_step_outputs[dataset_name][0].keys()
            di = {}
            # log individual metrics for each dataset
            for feat in feats:
                curr_dataset_curr_feature = np.mean([v[feat] for v in self.validation_step_outputs[dataset_name]])
                di[f"val_mean_{feat}/{dataset_name}"] = curr_dataset_curr_feature
                total_features[feat].append(curr_dataset_curr_feature)
            for k, v in di.items():
                self.log(k, v, on_epoch=True, prog_bar="rank_1/" in k)
                
        # from pytorch_lightning.callbacks import ModelCheckpoint
        # for callback in self.trainer.callbacks:
        #     if isinstance(callback, ModelCheckpoint):
        #         print(f"Checkpoint tracking {callback.monitor} -> Best model path: {callback.best_model_path}")
                
        # # log the avg metric for all datasets
        # for k, v in total_features.items():
        #     self.log(f"val_mean_{k}/all_nmr_combination_avg", np.mean(v), on_epoch=True, prog_bar="rank_1" in k)
            
        self.validation_step_outputs.clear()
        
        
    def on_test_epoch_end(self):
        # return
        total_features = defaultdict(list)
        for dataset_name in self.all_dataset_names:
            if self.test_step_outputs[dataset_name] == []:
                continue
            feats = self.test_step_outputs[dataset_name][0].keys()
            di = {}
            # log individual metrics for each dataset
            for feat in feats:
                curr_dataset_curr_feature = np.mean([v[feat] for v in self.test_step_outputs[dataset_name]])
                di[f"test_mean_{feat}/{dataset_name}"] = curr_dataset_curr_feature
                total_features[feat].append(curr_dataset_curr_feature)
            for k, v in di.items():
                self.log(k, v, on_epoch=True, prog_bar="rank_1" in k)
            
            for np_class, rank1_hits in self.test_np_classes_rank1[dataset_name].items():
                self.log(f"test/rank_1_of_NP_class/{np_class}/{dataset_name}", np.mean(rank1_hits), on_epoch=True)

        # # log the avg metric for all datasets
        # for k, v in total_features.items():
        #     self.log(f"test_mean_{k}/all_nmr_combination_avg", np.mean(v), on_epoch=True, prog_bar="rank_1" in k)
            
        self.test_step_outputs.clear()
        
    def on_train_epoch_end(self):
        if self.separate_classifier:
            for i, classifier in enumerate(self.classifiers):
                self.log(f"tr/weight_{self.all_dataset_names[i]}", classifier.weight.mean().item())
                self.log(f"tr/bias_{self.all_dataset_names[i]}", classifier.bias.mean().item())