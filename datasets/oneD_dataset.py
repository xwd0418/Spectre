"""Deprecated. Using this file may introduce some bug...."""

import logging
import pickle
import torch, os, pytorch_lightning as pl
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from datasets.hsqc_folder_dataset import get_delimeter, pad
from datasets.dataset_utils import specific_radius_mfp_loader
import sys, pathlib
repo_path = pathlib.Path(__file__).resolve().parents[1]


class OneDDataset(Dataset):
    '''
        Creates a folder-based dataset. Assumes that folder has the following structure: 

        {dir}/{[train, val, or test]}/HSQC
        - 0.pt
        - 1.pt
        - ...

        {dir}/{[train, val, or test]}/{FP | HYUN_FP}
        - 0.pt
        - 1.pt
        - ...
        
    '''
    def __init__(self, dir, split="train", FP_choice="", parser_args=None):
        self.dir = os.path.join(dir, split)
        self.dir_1d = f"/workspace/OneD_Only_Dataset/{split}"
        self.split = split
        self.fp_suffix = FP_choice
        self.parser_args = parser_args

        print(self.dir)
        assert(os.path.exists(self.dir))
        assert(split in ["train", "val", "test"])
        
        if parser_args['use_MW']:
            self.mol_weight_2d = pickle.load(open(os.path.join(self.dir, "MW/index.pkl"), 'rb'))
            self.mol_weight_1d = pickle.load(open(os.path.join(self.dir_1d, "MW/index.pkl"), 'rb'))
        self.files = os.listdir(os.path.join(self.dir, "oneD_NMR"))
        self.files_1d = os.listdir(os.path.join(self.dir_1d, "oneD_NMR"))

        logger = logging.getLogger("lightning")
        if dist.is_initialized():
            rank = dist.get_rank()
            if rank != 0:
                # For any process with rank other than 0, set logger level to WARNING or higher
                logger.setLevel(logging.WARNING)
        logger.info(f"[OneD Dataset]: dir={dir}, split={split},FP={FP_choice}")
        
        if parser_args['train_on_all_info_set']  or split in ["val", "test"]:
            logger.info(f"[OneD Dataset]: only all info datasets: {split}")
            path_to_load_full_info_indices = f"{repo_path}/datasets/{split}_indices_of_full_info_NMRs.pkl"
            self.files = pickle.load(open(path_to_load_full_info_indices, "rb"))
            logger.info(f"[OneD Dataset]: dataset size is {len(self)}")
            return 
        if self.parser_args['only_C_NMR']:
            def filter_unavailable_1d(x):
                c_tensor, h_tensor = torch.load(f"{self.dir_1d}/oneD_NMR/{x}")
                return len(c_tensor)>0
            def filter_unavailable(x):
                c_tensor, h_tensor = torch.load(f"{self.dir}/oneD_NMR/{x}")
                return len(c_tensor)>0
        elif self.parser_args['only_H_NMR']:
            def filter_unavailable_1d(x):
                c_tensor, h_tensor = torch.load(f"{self.dir_1d}/oneD_NMR/{x}")
                return len(h_tensor)>0
            def filter_unavailable(x):
                c_tensor, h_tensor = torch.load(f"{self.dir}/oneD_NMR/{x}")
                return len(h_tensor)>0
        else: # both 1d
            def filter_unavailable_1d(x):
                c_tensor, h_tensor = torch.load(f"{self.dir_1d}/oneD_NMR/{x}")
                return len(c_tensor)>0 and len(h_tensor)>0
            def filter_unavailable(x):
                c_tensor, h_tensor = torch.load(f"{self.dir}/oneD_NMR/{x}")
                return len(c_tensor)>0 and len(h_tensor)>0
        
        self.files = list(filter(filter_unavailable, self.files))
        self.files_1d = list(filter(filter_unavailable_1d, self.files_1d))
        logger.info(f"[OneD Dataset]: dataset size is {len(self)}")
        
    def __len__(self):
        # return 100
        length = len(self.files)
        if self.parser_args['train_on_all_info_set']:
           return length 
        length += len(self.files_1d)
        return length
        


    def __getitem__(self, idx):
        hsqc = torch.empty(0,3)
        
        if idx >= len(self.files): # load 1D dataset
            i = idx - len(self.files)
            # hsqc is empty tensor   
            c_tensor, h_tensor = torch.load(f"{self.dir_1d}/oneD_NMR/{self.files_1d[i]}")
            if self.parser_args['jittering'] == "normal" and self.split=="train":
                c_tensor = c_tensor + torch.randn_like(c_tensor) 
                h_tensor = h_tensor + torch.randn_like(h_tensor) * 0.1
            # No need to use optional inputs for 1D dataset 
            
        else :
            ### BEGINNING 2D dataset case
            i = idx
            c_tensor, h_tensor = torch.load(f"{self.dir}/oneD_NMR/{self.files[i]}")  
            if self.parser_args['jittering'] == "normal" and self.split=="train":
                c_tensor = c_tensor + torch.randn_like(c_tensor) 
                h_tensor = h_tensor + torch.randn_like(h_tensor) * 0.1
            # Again, no need to use optional inputs for 1D dataset
            ### ENDING 2D dataset case
            
        if self.parser_args['only_C_NMR']:
            h_tensor = torch.tensor([]) 
        elif self.parser_args['only_H_NMR']:
            c_tensor = torch.tensor([])
      
        c_tensor, h_tensor = c_tensor.view(-1, 1), h_tensor.view(-1, 1)
        c_tensor,h_tensor = F.pad(c_tensor, (0, 2), "constant", 0), F.pad(h_tensor, (0, 2), "constant", 0)     
            
        inputs = torch.vstack([
            get_delimeter("HSQC_start"),  hsqc,     get_delimeter("HSQC_end"),
            get_delimeter("C_NMR_start"), c_tensor, get_delimeter("C_NMR_end"), 
            get_delimeter("H_NMR_start"), h_tensor, get_delimeter("H_NMR_end"),
            ])    
                
            
        # loading MW and MFP in different datasets 
        if idx >= len(self.files): # load 1D dataset    
            mol_weight_dict = self.mol_weight_1d
            dataset_files = self.files_1d
            dataset_dir = self.dir_1d
            current_dataset = "1d"
        else:
            mol_weight_dict = self.mol_weight_2d
            dataset_files = self.files
            dataset_dir = self.dir
            current_dataset = "2d"
            
        if self.parser_args['use_MW']:
            mol_weight = mol_weight_dict[int(dataset_files[i].split(".")[0])]
            mol_weight = torch.tensor([mol_weight,0,0]).float()
            inputs = torch.vstack([inputs, get_delimeter("ms_start"), mol_weight, get_delimeter("ms_end")])
            
        if self.fp_suffix.startswith("pick_entropy"): # should be in the format of "pick_entropy_r9"
            mfp = specific_radius_mfp_loader.build_mfp(int(dataset_files[i].split(".")[0]), current_dataset ,self.split)
            # mfp_orig = torch.load(f"{dataset_dir}/R0_to_R4_reduced_FP/{dataset_files[i]}").float() 
            # print("current dataset is ", current_dataset)
            # print("load path is ", f"{dataset_dir}/R0_to_R4_reduced_FP/{dataset_files[i]}") 
            # print("i is ", i, "split is ", self.split)
            # assert (mfp==mfp_orig).all(), f"mfp should be the same\n mfp is " #{mfp.nonzero()}\n mfp_orig is {mfp_orig.nonzero()}"
        else:   
            mfp = torch.load(f"{dataset_dir}/{self.fp_suffix}/{dataset_files[i]}").float()  
     
        combined = (inputs, mfp)
        
        if self.parser_args['separate_classifier']:
            # input types are one of the following:
            # ["all_inputs", "HSQC_H_NMR", "HSQC_C_NMR", "only_hsqc", "only_1d", "only_H_NMR",  "only_C_NMR"]
            input_type = 0
            if len(hsqc):
                input_type+=4
            if len(h_tensor):
                input_type+=2
            if len(c_tensor):
                input_type+=1
            input_type = 7-input_type
            combined = (inputs, mfp, input_type)

        return combined
   


    

class  OneDDataModule(pl.LightningDataModule):
    def __init__(self, dir, FP_choice, batch_size: int = 32, parser_args=None):
        super().__init__()
        self.batch_size = batch_size
        self.dir = dir
        self.FP_choice = FP_choice
        self.collate_fn = pad
        self.parser_args = parser_args
    
    def setup(self, stage):
        if stage == "fit" or stage == "validate" or stage is None:
            self.train = OneDDataset(dir=self.dir, FP_choice=self.FP_choice, split="train", parser_args=self.parser_args)
            self.val = OneDDataset(dir=self.dir, FP_choice=self.FP_choice, split="val", parser_args=self.parser_args)
        if stage == "test":
            self.test = OneDDataset(dir=self.dir, FP_choice=self.FP_choice, split="test", parser_args=self.parser_args)
        if stage == "predict":
            raise NotImplementedError("Predict setup not implemented")

    def train_dataloader(self):
        return DataLoader(self.train, shuffle=True, batch_size=self.batch_size, collate_fn=self.collate_fn, 
                          num_workers=self.parser_args['num_workers'], pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, collate_fn=self.collate_fn, 
                          num_workers=self.parser_args['num_workers'], pin_memory=True, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, collate_fn=self.collate_fn, 
                          num_workers=self.parser_args['num_workers'], pin_memory=True, persistent_workers=True)