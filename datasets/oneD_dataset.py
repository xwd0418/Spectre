"""Used for training with all data of H | C | H and C"""

import logging
import pickle
import torch, os, pytorch_lightning as pl
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from datasets.hsqc_folder_dataset import pad, FolderDataset


import sys, pathlib
repo_path = pathlib.Path(__file__).resolve().parents[1]


class OneDDataset(FolderDataset):
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
    def __init__(self, dir, split="train", FP_choice="", parser_args=None, fp_loader=None):
        self.fp_loader = fp_loader
        
        self.dir = os.path.join(dir, split)
        self.dir_1d = f"/workspace/OneD_Only_Dataset/{split}"
        self.split = split
        self.fp_suffix = FP_choice
        self.parser_args = parser_args

        print(self.dir_1d)
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
        
        if split in ["test"]:
            with open(os.path.join(self.dir, "Superclass/index.pkl"), "rb") as f:
                self.NP_classes = pickle.load(f)
        
        if parser_args['train_on_all_info_set']  or split in ["val", "test"]:
            logger.info(f"[OneD Dataset]: only all info datasets: {split}")
            path_to_load_full_info_indices = f"{repo_path}/datasets/{split}_indices_of_full_info_NMRs.pkl"
            self.files = pickle.load(open(path_to_load_full_info_indices, "rb"))
            self.files.sort()
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
        length = len(self.files)
        if self.parser_args['train_on_all_info_set']  or self.split in ["val", "test"] :
           return length 
        length += len(self.files_1d)
        return length
        


    def __getitem__(self, idx):
        hsqc = torch.empty(0,3)
        
        # No need to use optional inputs for 1D dataset 
        if idx >= len(self.files): # load 1D dataset
            i = idx - len(self.files)
            # hsqc is empty tensor   
            c_tensor, h_tensor = torch.load(f"{self.dir_1d}/oneD_NMR/{self.files_1d[i]}")
            
            
        else :
            ### BEGINNING 2D dataset case
            i = idx
            c_tensor, h_tensor = torch.load(f"{self.dir}/oneD_NMR/{self.files[i]}")  
            
        if self.parser_args['jittering'] >0 and self.split=="train":
            jittering = self.parser_args['jittering']
            c_tensor = c_tensor + torch.randn_like(c_tensor) * jittering
            h_tensor = h_tensor + torch.randn_like(h_tensor) * jittering * 0.1
            
        if self.parser_args['only_C_NMR']:
            h_tensor = torch.tensor([]) 
        elif self.parser_args['only_H_NMR']:
            c_tensor = torch.tensor([])
      
        # loading MW and MFP in different datasets 
        if idx >= len(self.files): # load 1D dataset    
            if self.parser_args['use_MW']:
                mol_weight_dict = self.mol_weight_1d
            dataset_files = self.files_1d
            dataset_dir = self.dir_1d
            current_dataset = "1d"
        else:
            if self.parser_args['use_MW']:
                mol_weight_dict = self.mol_weight_2d
            dataset_files = self.files
            dataset_dir = self.dir
            current_dataset = "2d"
            
        mol_weight = None
        if self.parser_args['use_MW']:
            mol_weight = mol_weight_dict[int(dataset_files[i].split(".")[0])]
            mol_weight = torch.tensor([mol_weight,0,0]).float()
            
        # padding and stacking： 
        inputs, NMR_type_indicator = self.pad_and_stack_input(hsqc, c_tensor, h_tensor, mol_weight)
         
            
        if self.fp_suffix.startswith("pick_entropy") or self.fp_suffix.startswith("Morgan_FP") or self.fp_suffix.startswith("DB_specific_FP") or self.fp_suffix.startswith("Hash_Entropy"):
            mfp = self.fp_loader.build_mfp(int(dataset_files[i].split(".")[0]), current_dataset ,self.split)
            # mfp_orig = torch.load(f"{dataset_dir}/R0_to_R4_reduced_FP/{dataset_files[i]}").float() 
            # print("current dataset is ", current_dataset)
            # print("load path is ", f"{dataset_dir}/R0_to_R4_reduced_FP/{dataset_files[i]}") 
            # print("i is ", i, "split is ", self.split)
            # assert (mfp==mfp_orig).all(), f"mfp should be the same\n mfp is " #{mfp.nonzero()}\n mfp_orig is {mfp_orig.nonzero()}"
        else:   
            mfp = torch.load(f"{dataset_dir}/{self.fp_suffix}/{dataset_files[i]}").float()  
                
        combined = (inputs, mfp, NMR_type_indicator)
        
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
            combined = (inputs, mfp, NMR_type_indicator, mol_weight)
            
         
        if self.split in ["test"]:
            combined = (inputs, mfp, NMR_type_indicator, self.NP_classes[int(dataset_files[i].split(".")[0])])
        
        return combined
   


    

class  OneDDataModule(pl.LightningDataModule):
    def __init__(self, dir, FP_choice, fp_loader, batch_size: int = 32, parser_args=None):
        super().__init__()
        self.batch_size = batch_size
        self.dir = dir
        self.FP_choice = FP_choice
        self.collate_fn = pad
        self.parser_args = parser_args
        self.fp_loader = fp_loader
    
    def setup(self, stage):
        if stage == "fit" or stage == "validate" or stage is None:
            self.train = OneDDataset(dir=self.dir, FP_choice=self.FP_choice, split="train", parser_args=self.parser_args, fp_loader=self.fp_loader)
            self.val = OneDDataset(dir=self.dir, FP_choice=self.FP_choice, split="val", parser_args=self.parser_args, fp_loader=self.fp_loader)
        if stage == "test":
            self.test = OneDDataset(dir=self.dir, FP_choice=self.FP_choice, split="test", parser_args=self.parser_args, fp_loader=self.fp_loader)
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