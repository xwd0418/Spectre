
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch.utils.data import DataLoader, Dataset
from datasets.hsqc_folder_dataset import FolderDataset, FolderDataModule, pad
import torch, os, pytorch_lightning as pl, glob
import torch.nn.functional as F
import pickle

import sys, pathlib
repo_path = pathlib.Path(__file__).resolve().parents[1]



class All_Info_Dataset(FolderDataset):
    def __init__(self, dir, FP_choice, split, oneD_type, parser_args, has_HSQC = False, show_smiles=False, fp_loader=None):
        self.fp_loader = fp_loader
        
        self.dir = os.path.join(dir, split)
        self.fp_suffix = FP_choice
        self.split = split
        self.has_HSQC = has_HSQC
        self.parser_args = parser_args
        assert oneD_type in ["H_NMR", "C_NMR", "both", None]
        self.oneD_type = oneD_type
        path_to_load_full_info_indices = f"{repo_path}/datasets/{split}_indices_of_full_info_NMRs.pkl"
        self.files = pickle.load(open(path_to_load_full_info_indices, "rb"))
        self.files.sort()
        self.show_smiles = show_smiles
        if self.show_smiles:
            self.index_to_chemical_names = pickle.load(open(f'/workspace/SMILES_dataset/{split}/Chemical/index.pkl', 'rb'))
            self.index_to_smiles = pickle.load(open(f'/workspace/SMILES_dataset/{split}/SMILES/index.pkl', 'rb'))
            
        if parser_args['use_MW']:
            self.mol_weight = pickle.load(open(os.path.join(self.dir, "MW/index.pkl"), 'rb'))

        if split in ["test"]:
            with open(os.path.join(self.dir, "Superclass/index.pkl"), "rb") as f:
                self.NP_classes = pickle.load(f)
        
    def __len__(self):
        # if self.parser_args['debug'] or self.parser_args['foldername'] == 'debug':
        #     return 500
        # return 200
        return len(self.files)
    
    def __getitem__(self, i):
        NMR_path= f"{self.dir}/oneD_NMR/{self.files[i]}"
        
        if self.oneD_type == None:
            c_tensor = torch.tensor([]).view(-1, 1)
            h_tensor = torch.tensor([]).view(-1, 1)
        else:
            c_tensor, h_tensor = torch.load(f"{self.dir}/oneD_NMR/{self.files[i]}")   # both
            if self.parser_args['jittering'] == "normal" and self.split=="train":
                c_tensor = c_tensor + torch.randn_like(c_tensor) 
                h_tensor = h_tensor + torch.randn_like(h_tensor) * 0.1
            if self.oneD_type == "H_NMR":
                c_tensor = torch.tensor([]).view(-1, 1)
            elif self.oneD_type == "C_NMR":
                h_tensor = torch.tensor([]).view(-1, 1)
         
        if self.has_HSQC:
            hsqc = torch.load(f"{self.dir}/HSQC/{self.files[i]}").float()
            if self.parser_args['jittering'] == "normal" and self.split=="train":
                    hsqc[:,0] = hsqc[:,0] + torch.randn_like(hsqc[:,0]) 
                    hsqc[:,1] = hsqc[:,1] + torch.randn_like(hsqc[:,1]) * 0.1
        else:
            hsqc =  torch.empty(0,3)
        
        if len(hsqc)==len(c_tensor)==len(h_tensor)==0:
            exit(f"Error: {self.files[i]} has no data")
            return None
        
        mol_weight = None
        if self.parser_args['use_MW']:
            mol_weight = self.mol_weight[int(self.files[i].split(".")[0])]
            mol_weight = torch.tensor([mol_weight,0,0]).float()
        
        # padding and stacking： 
        inputs, NMR_type_indicator = self.pad_and_stack_input(hsqc, c_tensor, h_tensor, mol_weight)
         
        file_index = int(self.files[i].split(".")[0])
        if self.show_smiles: # prediction stage
            smiles = self.index_to_smiles[file_index]
            chemical_name = self.index_to_chemical_names[file_index]
            if self.split in ["test"]:
                return inputs, (smiles, chemical_name, NMR_type_indicator, NMR_path, self.NP_classes[file_index])
            return inputs, (smiles, chemical_name, NMR_type_indicator, NMR_path)
        
        if self.fp_suffix.startswith("pick_entropy") or self.fp_suffix.startswith("Morgan_FP") or self.fp_suffix.startswith("DB_specific_FP") or self.fp_suffix.startswith("Hash_Entropy"): # should be in the format of "pick_entropy_r9"
            mfp = self.fp_loader.build_mfp(int(self.files[i].split(".")[0]), "2d" ,self.split)
        else:   
            mfp = torch.load(f"{self.dir}/{self.fp_suffix}/{self.files[i]}")


        if self.parser_args['loss_func'] == "CE":
            num_class = self.parser_args['num_class']
            mfp = torch.where(mfp >= num_class, num_class-1, mfp).long()

        if self.split in ["test"]:
            return (inputs, mfp, NMR_type_indicator, self.NP_classes[file_index])
    
        return (inputs, mfp, NMR_type_indicator)



class OptionalInputDataModule(FolderDataModule):
    def __init__(self, dir, FP_choice, input_src, fp_loader, batch_size: int = 32, parser_args=None):
        super().__init__(dir, FP_choice, input_src, fp_loader, batch_size, parser_args)
        self.fp_loader = fp_loader
        self.collate_fn = collate_to_filter_bad_sample_and_pad
        
    def setup(self, stage):
        if stage == "fit" or stage == "validate" or stage is None:
            self.train = FolderDataset(dir=self.dir, FP_choice=self.FP_choice, input_src = self.input_src, split="train", parser_args=self.parser_args,  fp_loader=self.fp_loader)
            
            self.val_all_inputs = All_Info_Dataset(dir=self.dir, FP_choice=self.FP_choice, split="val", oneD_type="both",  parser_args=self.parser_args, has_HSQC=True, fp_loader=self.fp_loader)
            self.val_only_hsqc  = All_Info_Dataset(dir=self.dir, FP_choice=self.FP_choice, split="val", oneD_type=None,    parser_args=self.parser_args, has_HSQC=True, fp_loader=self.fp_loader)
            self.val_only_1d    = All_Info_Dataset(dir=self.dir, FP_choice=self.FP_choice, split="val", oneD_type="both",  parser_args=self.parser_args, fp_loader=self.fp_loader)
            self.val_only_H_NMR = All_Info_Dataset(dir=self.dir, FP_choice=self.FP_choice, split="val", oneD_type="H_NMR", parser_args=self.parser_args, fp_loader=self.fp_loader)
            self.val_HSQC_H_NMR = All_Info_Dataset(dir=self.dir, FP_choice=self.FP_choice, split="val", oneD_type="H_NMR", parser_args=self.parser_args, has_HSQC=True, fp_loader=self.fp_loader)
            self.val_only_C_NMR = All_Info_Dataset(dir=self.dir, FP_choice=self.FP_choice, split="val", oneD_type="C_NMR", parser_args=self.parser_args, fp_loader=self.fp_loader)
            self.val_HSQC_C_NMR = All_Info_Dataset(dir=self.dir, FP_choice=self.FP_choice, split="val", oneD_type="C_NMR", parser_args=self.parser_args, has_HSQC=True, fp_loader=self.fp_loader)

        if stage == "test":
            self.test_all_inputs = All_Info_Dataset(dir=self.dir, FP_choice=self.FP_choice, split="test", oneD_type="both",  parser_args=self.parser_args, has_HSQC=True, fp_loader=self.fp_loader)
            self.test_only_hsqc  = All_Info_Dataset(dir=self.dir, FP_choice=self.FP_choice, split="test", oneD_type=None,    parser_args=self.parser_args, has_HSQC=True, fp_loader=self.fp_loader)
            self.test_only_1d    = All_Info_Dataset(dir=self.dir, FP_choice=self.FP_choice, split="test", oneD_type="both",  parser_args=self.parser_args, fp_loader=self.fp_loader)
            self.test_only_H_NMR = All_Info_Dataset(dir=self.dir, FP_choice=self.FP_choice, split="test", oneD_type="H_NMR", parser_args=self.parser_args, fp_loader=self.fp_loader)
            self.test_HSQC_H_NMR = All_Info_Dataset(dir=self.dir, FP_choice=self.FP_choice, split="test", oneD_type="H_NMR", parser_args=self.parser_args, has_HSQC=True, fp_loader=self.fp_loader)
            self.test_only_C_NMR = All_Info_Dataset(dir=self.dir, FP_choice=self.FP_choice, split="test", oneD_type="C_NMR", parser_args=self.parser_args, fp_loader=self.fp_loader)
            self.test_HSQC_C_NMR = All_Info_Dataset(dir=self.dir, FP_choice=self.FP_choice, split="test", oneD_type="C_NMR", parser_args=self.parser_args, has_HSQC=True, fp_loader=self.fp_loader)
            
            
        if stage == "predict":
            self.predict_stage_all_inputs = All_Info_Dataset(dir=self.dir, FP_choice=self.FP_choice, split="test", oneD_type="both",  parser_args=self.parser_args, has_HSQC=True, show_smiles=True, fp_loader=self.fp_loader)
            self.predict_stage_only_hsqc  = All_Info_Dataset(dir=self.dir, FP_choice=self.FP_choice, split="test", oneD_type=None,    parser_args=self.parser_args, has_HSQC=True, show_smiles=True, fp_loader=self.fp_loader)
            self.predict_stage_only_1d    = All_Info_Dataset(dir=self.dir, FP_choice=self.FP_choice, split="test", oneD_type="both",  parser_args=self.parser_args, show_smiles=True, fp_loader=self.fp_loader)
            self.predict_stage_only_H_NMR = All_Info_Dataset(dir=self.dir, FP_choice=self.FP_choice, split="test", oneD_type="H_NMR", parser_args=self.parser_args, show_smiles=True, fp_loader=self.fp_loader)
            self.predict_stage_HSQC_H_NMR = All_Info_Dataset(dir=self.dir, FP_choice=self.FP_choice, split="test", oneD_type="H_NMR", parser_args=self.parser_args, has_HSQC=True, show_smiles=True, fp_loader=self.fp_loader)
            self.predict_stage_only_C_NMR = All_Info_Dataset(dir=self.dir, FP_choice=self.FP_choice, split="test", oneD_type="C_NMR", parser_args=self.parser_args, show_smiles=True, fp_loader=self.fp_loader)
            self.predict_stage_HSQC_C_NMR = All_Info_Dataset(dir=self.dir, FP_choice=self.FP_choice, split="test", oneD_type="C_NMR", parser_args=self.parser_args, has_HSQC=True, show_smiles=True, fp_loader=self.fp_loader)
            

    def train_dataloader(self):
        sampler = None
        should_shuffle = True
        if self.parser_args['weighted_sample_based_on_input_type']:
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=self.train.get_weight_of_samples_based_on_input_type(),
                num_samples=len(self.train),
                replacement=True
            )
            should_shuffle = False
        return DataLoader(self.train, shuffle=should_shuffle, batch_size=self.batch_size, collate_fn=self.collate_fn, sampler=sampler,
                          num_workers=self.parser_args['num_workers'], pin_memory=True, persistent_workers=self.parser_args['num_workers']>1)
        
    def val_dataloader(self):
        loader_all_inputs = DataLoader(self.val_all_inputs, batch_size=self.batch_size, collate_fn=self.collate_fn, 
                                      num_workers=self.parser_args['num_workers'], pin_memory=True, persistent_workers=self.parser_args['num_workers']>1)  
        loader_only_hsqc = DataLoader(self.val_only_hsqc, batch_size=self.batch_size, collate_fn=self.collate_fn, 
                                      num_workers=self.parser_args['num_workers'], pin_memory=True, persistent_workers=self.parser_args['num_workers']>1)
        loader_only_1d = DataLoader(self.val_only_1d, batch_size=self.batch_size, collate_fn=self.collate_fn, 
                                      num_workers=self.parser_args['num_workers'], pin_memory=True, persistent_workers=self.parser_args['num_workers']>1)
        loader_only_H_NMR = DataLoader(self.val_only_H_NMR, batch_size=self.batch_size, collate_fn=self.collate_fn, 
                                      num_workers=self.parser_args['num_workers'], pin_memory=True, persistent_workers=self.parser_args['num_workers']>1)
        loader_HSQC_H_NMR = DataLoader(self.val_HSQC_H_NMR, batch_size=self.batch_size, collate_fn=self.collate_fn,
                                        num_workers=self.parser_args['num_workers'], pin_memory=True, persistent_workers=self.parser_args['num_workers']>1)
        loader_only_C_NMR = DataLoader(self.val_only_C_NMR, batch_size=self.batch_size, collate_fn=self.collate_fn,
                                        num_workers=self.parser_args['num_workers'], pin_memory=True, persistent_workers=self.parser_args['num_workers']>1)
        loader_HSQC_C_NMR = DataLoader(self.val_HSQC_C_NMR, batch_size=self.batch_size, collate_fn=self.collate_fn,
                                        num_workers=self.parser_args['num_workers'], pin_memory=True, persistent_workers=self.parser_args['num_workers']>1)
        return [loader_all_inputs, loader_HSQC_H_NMR, loader_HSQC_C_NMR, loader_only_hsqc, loader_only_1d, loader_only_H_NMR, loader_only_C_NMR]
    
    def test_dataloader(self):
        loader_all_inputs = DataLoader(self.test_all_inputs, batch_size=self.batch_size, collate_fn=self.collate_fn, 
                                      num_workers=self.parser_args['num_workers'], pin_memory=True, persistent_workers=self.parser_args['num_workers']>1)
        loader_only_hsqc = DataLoader(self.test_only_hsqc, batch_size=self.batch_size, collate_fn=self.collate_fn,  
                                        num_workers=self.parser_args['num_workers'], pin_memory=True, persistent_workers=self.parser_args['num_workers']>1)
        loader_only_1d = DataLoader(self.test_only_1d, batch_size=self.batch_size, collate_fn=self.collate_fn,
                                        num_workers=self.parser_args['num_workers'], pin_memory=True, persistent_workers=self.parser_args['num_workers']>1)
        loader_only_H_NMR = DataLoader(self.test_only_H_NMR, batch_size=self.batch_size, collate_fn=self.collate_fn,
                                        num_workers=self.parser_args['num_workers'], pin_memory=True, persistent_workers=self.parser_args['num_workers']>1)
        loader_HSQC_H_NMR = DataLoader(self.test_HSQC_H_NMR, batch_size=self.batch_size, collate_fn=self.collate_fn,
                                        num_workers=self.parser_args['num_workers'], pin_memory=True, persistent_workers=self.parser_args['num_workers']>1)
        loader_only_C_NMR = DataLoader(self.test_only_C_NMR, batch_size=self.batch_size, collate_fn=self.collate_fn,
                                        num_workers=self.parser_args['num_workers'], pin_memory=True, persistent_workers=self.parser_args['num_workers']>1)
        loader_HSQC_C_NMR = DataLoader(self.test_HSQC_C_NMR, batch_size=self.batch_size, collate_fn=self.collate_fn,
                                        num_workers=self.parser_args['num_workers'], pin_memory=True, persistent_workers=self.parser_args['num_workers']>1)
        return [loader_all_inputs, loader_HSQC_H_NMR, loader_HSQC_C_NMR, loader_only_hsqc, loader_only_1d, loader_only_H_NMR, loader_only_C_NMR]
      
    def predict_dataloader(self, shuffle = False):
        loader_all_inputs = DataLoader(self.predict_stage_all_inputs, batch_size=self.batch_size, collate_fn=self.collate_fn, shuffle = shuffle,
                                        num_workers=self.parser_args['num_workers'], pin_memory=True, persistent_workers=self.parser_args['num_workers']>1)
        loader_only_hsqc = DataLoader(self.predict_stage_only_hsqc, batch_size=self.batch_size, collate_fn=self.collate_fn, shuffle = shuffle,
                                        num_workers=self.parser_args['num_workers'], pin_memory=True, persistent_workers=self.parser_args['num_workers']>1)
        loader_only_1d = DataLoader(self.predict_stage_only_1d, batch_size=self.batch_size, collate_fn=self.collate_fn, shuffle = shuffle,

                                        num_workers=self.parser_args['num_workers'], pin_memory=True, persistent_workers=self.parser_args['num_workers']>1)
        loader_only_H_NMR = DataLoader(self.predict_stage_only_H_NMR, batch_size=self.batch_size, collate_fn=self.collate_fn, shuffle = shuffle,
                                        num_workers=self.parser_args['num_workers'], pin_memory=True, persistent_workers=self.parser_args['num_workers']>1)
        loader_HSQC_H_NMR = DataLoader(self.predict_stage_HSQC_H_NMR, batch_size=self.batch_size, collate_fn=self.collate_fn, shuffle = shuffle,
                                        num_workers=self.parser_args['num_workers'], pin_memory=True, persistent_workers=self.parser_args['num_workers']>1)
        loader_only_C_NMR = DataLoader(self.predict_stage_only_C_NMR, batch_size=self.batch_size, collate_fn=self.collate_fn, shuffle = shuffle,
                                        num_workers=self.parser_args['num_workers'], pin_memory=True, persistent_workers=self.parser_args['num_workers']>1)
        loader_HSQC_C_NMR = DataLoader(self.predict_stage_HSQC_C_NMR, batch_size=self.batch_size, collate_fn=self.collate_fn, shuffle = shuffle,
                                        num_workers=self.parser_args['num_workers'], pin_memory=True, persistent_workers=self.parser_args['num_workers']>1)
        return [loader_all_inputs, loader_HSQC_H_NMR, loader_HSQC_C_NMR, loader_only_hsqc, loader_only_1d, loader_only_H_NMR, loader_only_C_NMR]



# used as collate_fn in the dataloader
def collate_to_filter_bad_sample_and_pad(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:  # Important to avoid passing an empty batch
        return None
    return pad(batch) # pad the batch to the same length