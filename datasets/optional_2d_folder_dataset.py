
from torch.utils.data import DataLoader, Dataset
from datasets.hsqc_folder_dataset import FolderDataset, FolderDataModule, pad, get_delimeter
import torch, os, pytorch_lightning as pl, glob
import torch.nn.functional as F


'''
OneD Dataset, only for evaluting optional input
'''
class OneD_Dataset(Dataset):
    def __init__(self, dir, FP_choice, split, oneD_type, parser_args):
        self.dir = os.path.join(dir, split)
        self.fp_suffix = FP_choice
        self.split = split
        self.parser_args = parser_args
        assert oneD_type in ["H_NMR", "C_NMR", "both"]
        self.oneD_type = oneD_type
        self.files = os.listdir(os.path.join(self.dir, "oneD_NMR"))
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, i):
        c_tensor, h_tensor = torch.load(f"{self.dir}/oneD_NMR/{self.files[i]}")  
        if self.oneD_type == "H_NMR":
            c_tensor = torch.tensor([]).view(-1, 1)
        elif self.oneD_type == "C_NMR":
            h_tensor = torch.tensor([]).view(-1, 1)
         
        hsqc =  torch.empty(0,3)
        if len(hsqc)==len(c_tensor)==len(h_tensor)==0:
            return None
        c_tensor, h_tensor = c_tensor.view(-1, 1), h_tensor.view(-1, 1)
        c_tensor,h_tensor = F.pad(c_tensor, (0, 2), "constant", 0), F.pad(h_tensor, (0, 2), "constant", 0)
        inputs = torch.vstack([
                get_delimeter("HSQC_start"),  hsqc,     get_delimeter("HSQC_end"),
                get_delimeter("C_NMR_start"), c_tensor, get_delimeter("C_NMR_end"), 
                get_delimeter("H_NMR_start"), h_tensor, get_delimeter("H_NMR_end"),
                ])   
        mfp = torch.load(f"{self.dir}/{self.fp_suffix}/{self.files[i]}").float()  
        if self.parser_args['loss_func'] == "CE":
            num_class = self.parser_args['num_class']
            mfp = torch.where(mfp >= num_class, num_class-1, mfp).long()
        return (inputs, mfp)



class OptionalInputDataModule(FolderDataModule):
    def __init__(self, dir, FP_choice, input_src, batch_size: int = 32, parser_args=None):
        super().__init__(dir, FP_choice, input_src, batch_size, parser_args)
        self.collate_fn = collate_to_filter_bad_sample_and_pad
        
    def setup(self, stage):
        if stage == "fit" or stage == "validate" or stage is None:
            self.train = FolderDataset(dir=self.dir, FP_choice=self.FP_choice, input_src = self.input_src, split="train", parser_args=self.parser_args)
            
            self.val_all_inputs = FolderDataset(dir=self.dir, FP_choice=self.FP_choice, input_src = self.input_src,split="val", parser_args=self.parser_args)
            self.parser_args['enable_hsqc_delimeter_only_2d'] = True
            self.val_only_hsqc = FolderDataset(dir=self.dir, FP_choice=self.FP_choice, input_src = ["HSQC"], split="val", parser_args=self.parser_args)
            self.parser_args['enable_hsqc_delimeter_only_2d'] = False
            self.val_only_1d = OneD_Dataset(dir=self.dir, FP_choice=self.FP_choice, split="val", oneD_type="both", parser_args=self.parser_args)
            self.val_only_H_NMR = OneD_Dataset(dir=self.dir, FP_choice=self.FP_choice, split="val", oneD_type="H_NMR", parser_args=self.parser_args)
            self.val_only_C_NMR = OneD_Dataset(dir=self.dir, FP_choice=self.FP_choice, split="val", oneD_type="C_NMR", parser_args=self.parser_args)
            
        if stage == "test":
            self.test_all_inputs = FolderDataset(dir=self.dir, FP_choice=self.FP_choice, input_src = self.input_src, split="test", parser_args=self.parser_args)
            self.parser_args['enable_hsqc_delimeter_only_2d'] = True
            self.test_only_hsqc = FolderDataset(dir=self.dir, FP_choice=self.FP_choice, input_src = ["HSQC"], split="test", parser_args=self.parser_args)
            self.parser_args['enable_hsqc_delimeter_only_2d'] = False
            self.test_only_1d = OneD_Dataset(dir=self.dir, FP_choice=self.FP_choice, split="test", oneD_type="both", parser_args=self.parser_args)
            self.test_only_H_NMR = OneD_Dataset(dir=self.dir, FP_choice=self.FP_choice, split="test", oneD_type="H_NMR", parser_args=self.parser_args)
            self.test_only_C_NMR = OneD_Dataset(dir=self.dir, FP_choice=self.FP_choice, split="test", oneD_type="C_NMR", parser_args=self.parser_args)
            
        if stage == "predict":
            raise NotImplementedError("Predict setup not implemented")

    def train_dataloader(self):
        return DataLoader(self.train, shuffle=True, batch_size=self.batch_size, collate_fn=self.collate_fn, 
                          num_workers=self.parser_args['num_workers'], pin_memory=True, persistent_workers=True)
        
    def val_dataloader(self):
        loader_all_inputs = DataLoader(self.val_all_inputs, batch_size=self.batch_size, collate_fn=self.collate_fn, 
                                      num_workers=self.parser_args['num_workers'], pin_memory=True, persistent_workers=True)  
        loader_only_hsqc = DataLoader(self.val_only_hsqc, batch_size=self.batch_size, collate_fn=self.collate_fn, 
                                      num_workers=self.parser_args['num_workers'], pin_memory=True, persistent_workers=True)
        loader_only_1d = DataLoader(self.val_only_1d, batch_size=self.batch_size, collate_fn=self.collate_fn, 
                                      num_workers=self.parser_args['num_workers'], pin_memory=True, persistent_workers=True)
        loader_only_H_NMR = DataLoader(self.val_only_H_NMR, batch_size=self.batch_size, collate_fn=self.collate_fn, 
                                      num_workers=self.parser_args['num_workers'], pin_memory=True, persistent_workers=True)
        loader_only_C_NMR = DataLoader(self.val_only_C_NMR, batch_size=self.batch_size, collate_fn=self.collate_fn,
                                        num_workers=self.parser_args['num_workers'], pin_memory=True, persistent_workers=True)
        return [loader_all_inputs, loader_only_hsqc, loader_only_1d, loader_only_H_NMR, loader_only_C_NMR]
    
    def test_dataloader(self):
        loader_all_inputs = DataLoader(self.test_all_inputs, batch_size=self.batch_size, collate_fn=self.collate_fn, 
                                      num_workers=self.parser_args['num_workers'], pin_memory=True, persistent_workers=True)
        loader_only_hsqc = DataLoader(self.test_only_hsqc, batch_size=self.batch_size, collate_fn=self.collate_fn,  
                                        num_workers=self.parser_args['num_workers'], pin_memory=True, persistent_workers=True)
        loader_only_1d = DataLoader(self.test_only_1d, batch_size=self.batch_size, collate_fn=self.collate_fn,
                                        num_workers=self.parser_args['num_workers'], pin_memory=True, persistent_workers=True)
        loader_only_H_NMR = DataLoader(self.test_only_H_NMR, batch_size=self.batch_size, collate_fn=self.collate_fn,
                                        num_workers=self.parser_args['num_workers'], pin_memory=True, persistent_workers=True)
        loader_only_C_NMR = DataLoader(self.test_only_C_NMR, batch_size=self.batch_size, collate_fn=self.collate_fn,
                                        num_workers=self.parser_args['num_workers'], pin_memory=True, persistent_workers=True)
        return [loader_all_inputs, loader_only_hsqc, loader_only_1d, loader_only_H_NMR, loader_only_C_NMR]
      
# used as collate_fn in the dataloader
def collate_to_filter_bad_sample_and_pad(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:  # Important to avoid passing an empty batch
        return None
    return pad(batch) # pad the batch to the same length