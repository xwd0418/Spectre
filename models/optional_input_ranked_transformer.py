from models.ranked_transformer import HsqcRankedTransformer
from collections import defaultdict
import numpy as np

class OptionalInputRankedTransformer(HsqcRankedTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.validation_step_outputs = defaultdict(list)
        self.test_step_outputs = defaultdict(list)
        self.all_dataset_names = ["all_inputs", "only_hsqc", "only_1d", "only_H_NMR", "only_C_NMR"]
         
    def validation_step(self, batch, batch_idx, dataloader_idx ):
        
        current_batch_name = self.all_dataset_names[dataloader_idx]
        metrics = super().validation_step(batch, batch_idx)
        self.validation_step_outputs[current_batch_name].append(metrics)
        return metrics
    
    def test_step(self, batch, batch_idx, dataloader_idx):
        
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
                di[f"val_{dataset_name}/mean_{feat}"] = np.mean([v[feat]
                                                 for v in self.validation_step_outputs[dataset_name]])
            for k, v in di.items():
                self.log(k, v, on_epoch=True, prog_bar=k=="val/mean_all_inputs_rank_1")
        self.validation_step_outputs.clear()
        
        
    def on_test_epoch_end(self):
        for dataset_name in self.all_dataset_names:
            feats = self.test_step_outputs[dataset_name][0].keys()
            di = {}
            for feat in feats:
                di[f"test_{dataset_name}/mean_{feat}"] = np.mean([v[feat]
                                                 for v in self.test_step_outputs[dataset_name]])
            for k, v in di.items():
                self.log(k, v, on_epoch=True)
        self.test_step_outputs.clear()