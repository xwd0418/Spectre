"""Used for training with all data of HSQC | HSQC+H | HSQC+C | all info | optional inputs"""


import logging
import pickle, random
import torch, os, pytorch_lightning as pl, glob
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F



import sys, pathlib
repo_path = pathlib.Path(__file__).resolve().parents[1]

class FolderDataset(Dataset):
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
    def __init__(self, dir, split="train", input_src=["HSQC"], FP_choice="", parser_args=None):
        from datasets.dataset_utils import fp_loader_configer
        self.fp_loader = fp_loader_configer.fp_loader

        self.dir = os.path.join(dir, split)
        self.split = split
        self.fp_suffix = FP_choice
        self.input_src = input_src
        self.parser_args = parser_args
        logger = logging.getLogger("lightning")

        assert os.path.exists(self.dir), f"{self.dir} does not exist"
        assert(split in ["train", "val", "test"])
        for src in input_src:
            assert os.path.exists(os.path.join(self.dir, src)),"{} does not exist".format(os.path.join(self.dir, src))
            
        if parser_args['use_MW']:
            self.mol_weight_2d = pickle.load(open(os.path.join(self.dir, "MW/index.pkl"), 'rb'))
            
            
        if parser_args['train_on_all_info_set'] or split in ["val", "test"]:
            logger.info(f"[FolderDataset]: only all info datasets")
            path_to_load_full_info_indices = f"{repo_path}/datasets/{split}_indices_of_full_info_NMRs.pkl"
            self.files = pickle.load(open(path_to_load_full_info_indices, "rb"))
            print("loaded full info indices\n\n\n")
            # print("36690.pt" in self.files, self.files[0])
            # assert (not parser_args['combine_oneD_only_dataset'])
        else:    
            self.files = os.listdir(os.path.join(self.dir, "HSQC")) 
        self.files.sort() # sorted because we need to find correct weight mappings 
        if parser_args['combine_oneD_only_dataset']: # load 1D dataset as well 
            self.dir_1d = f"/workspace/OneD_Only_Dataset/{split}"
            
            self.mol_weight_1d = pickle.load(open(os.path.join(self.dir_1d, "MW/index.pkl"), 'rb'))
            self.files_1d = os.listdir(os.path.join(self.dir_1d, "oneD_NMR/"))
            self.files_1d.sort()
            
        if dist.is_initialized():
            rank = dist.get_rank()
            if rank != 0:
                # For any process with rank other than 0, set logger level to WARNING or higher
                logger.setLevel(logging.WARNING)
        logger.info(f"[FolderDataset]: dir={dir},input_src={input_src},split={split},FP={FP_choice},normalize_hsqc={parser_args['normalize_hsqc']}")
        
        if self.parser_args['only_C_NMR']:
            def filter_unavailable(x):
                if os.path.exists(os.path.join(self.dir, "oneD_NMR", x)) == False:
                    return False 
                c_tensor, h_tensor = torch.load(f"{self.dir}/oneD_NMR/{x}")
                return len(c_tensor)>0
            self.files = list(filter(filter_unavailable, self.files))
        elif self.parser_args['only_H_NMR']:
            def filter_unavailable(x):
                if os.path.exists(os.path.join(self.dir, "oneD_NMR", x)) == False:
                    return False 
                c_tensor, h_tensor = torch.load(f"{self.dir}/oneD_NMR/{x}")
                return len(h_tensor)>0
            self.files = list(filter(filter_unavailable, self.files))
            
        elif self.parser_args['only_oneD_NMR']:
            def filter_unavailable(x):
                if os.path.exists(os.path.join(self.dir, "oneD_NMR", x)) == False:
                    return False 
                c_tensor, h_tensor = torch.load(f"{self.dir}/oneD_NMR/{x}")
                return len(h_tensor)>0 and len(c_tensor)>0
            self.files = list(filter(filter_unavailable, self.files))

        logger.info(f"[FolderDataset]: dataset size is {len(self)}")
        
        # print("\n\n\n", "36690.pt" in self.files, self.files[0])
        
        
        
    def __len__(self):
        if self.parser_args['debug'] or self.parser_args['foldername'] == "debug":
            return 50000
        length = len(self.files)
        if self.parser_args['combine_oneD_only_dataset']:
            length += len(self.files_1d)
        return length
        


    def __getitem__(self, idx):
        
        if idx >= len(self.files): 
        ### load 1D dataset
            current_dataset = "1d"
            i = idx - len(self.files)
            # hsqc is empty tensor
            hsqc = torch.empty(0,3)
            c_tensor, h_tensor = torch.load(f"{self.dir_1d}/oneD_NMR/{self.files_1d[i]}")
            if self.parser_args['jittering'] == "normal" and self.split=="train":
                c_tensor = c_tensor + torch.randn_like(c_tensor)
                h_tensor = h_tensor + torch.randn_like(h_tensor) * 0.1
                
            # input dropout
            if self.parser_args['optional_inputs'] and len(c_tensor) > 0 and len(h_tensor) > 0:
                if not self.parser_args['combine_oneD_only_dataset'] :
                    raise NotImplementedError("optional_inputs is only supported when combine_oneD_only_dataset is True")
                random_num = random.random()
                if random_num <= 0.3984: # drop C rate 
                    c_tensor = torch.tensor([]) 
                elif random_num <= 0.3984+0.2032: # drop H rate
                    h_tensor = torch.tensor([])
                    
        else :
        ### BEGINNING 2D dataset case
            current_dataset = "2d"
            i = idx
            def file_exist(src, filename):
                return os.path.exists(os.path.join(self.dir, src, filename))
            
            # Load HSQC as 1-channel image 
            if "HSQC_images_1channel" in self.input_src:
                inputs = torch.load(f"{self.dir}/HSQC_images_1channel/{self.files[i]}")
            elif "HSQC_images_2channel" in self.input_src:
                inputs = torch.load(f"{self.dir}/HSQC_images_2channel/{self.files[i]}")
            
            # Load HSQC as sequence
            elif "HSQC" in self.input_src:
                hsqc = torch.load(f"{self.dir}/HSQC/{self.files[i]}").type(torch.FloatTensor)
                if self.parser_args['jittering'] == "normal" and self.split=="train":
                    hsqc[:,0] = hsqc[:,0] + torch.randn_like(hsqc[:,0]) 
                    hsqc[:,1] = hsqc[:,1] + torch.randn_like(hsqc[:,1]) * 0.1
            
                if self.parser_args['use_peak_values']:
                    hsqc = normalize_hsqc(hsqc)
                inputs = hsqc
            
            c_tensor, h_tensor = (torch.tensor([]) , torch.tensor([])) 
            if "oneD_NMR" in self.input_src:
                if file_exist("oneD_NMR", self.files[i]):
                    c_tensor, h_tensor = torch.load(f"{self.dir}/oneD_NMR/{self.files[i]}")  
                    if self.parser_args['jittering'] == "normal" and self.split=="train":
                        c_tensor = c_tensor + torch.randn_like(c_tensor) 
                        h_tensor = h_tensor + torch.randn_like(h_tensor) * 0.1
                    # randomly drop 1D and 2D NMRs if needed
                    if self.parser_args['optional_inputs']:
                        # DO NOT drop 2D, cuz we have enough amount of 1D data 
                        # if random.random() <= 1/2: 
                        #     hsqc =  torch.empty(0,3)
                        if len(c_tensor)>0 and len(h_tensor)>0:
                            # it is fine to drop one of the oneD NMRs
                            random_num_for_dropping =  random.random()                            
                            if random_num_for_dropping <= 0.3530:# drop C rate
                                c_tensor = torch.tensor([])
                            elif random_num_for_dropping <= 0.3530+0.2939: # drop H rate
                                h_tensor = torch.tensor([])
                            # else: keep both
                                          
                        assert (len(hsqc) > 0 or len(c_tensor) > 0 or len(h_tensor) > 0), "all NMRs are dropped"
                            
            if self.parser_args['only_oneD_NMR']:
                hsqc = torch.empty(0,3)                
            if self.parser_args['only_C_NMR']:
                h_tensor = torch.tensor([])
            if self.parser_args['only_H_NMR']:
                c_tensor = torch.tensor([])
        ### ENDING 2D dataset case
                
            
        # loading MW and MFP in different datasets 
        if idx >= len(self.files) : # load 1D dataset    
            if self.parser_args['use_MW']:
                mol_weight_dict = self.mol_weight_1d
            dataset_files = self.files_1d
            dataset_dir = self.dir_1d
        else:
            if self.parser_args['use_MW']:
                mol_weight_dict = self.mol_weight_2d
            dataset_files = self.files
            dataset_dir = self.dir
            
        mol_weight = None
        if self.parser_args['use_MW']:
            mol_weight = mol_weight_dict[int(dataset_files[i].split(".")[0])]
            mol_weight = torch.tensor([mol_weight,0,0]).float()
            
            if self.parser_args['optional_inputs']:
                if random.random() <= 0.5:
                    mol_weight = torch.tensor([])
                
            
        # padding and stacking： 
        inputs, NMR_type_indicator = self.pad_and_stack_input(hsqc, c_tensor, h_tensor, mol_weight)
            
        # remember build ranking set
        
        if self.fp_suffix.startswith("pick_entropy") or self.fp_suffix.startswith("DB_specific_FP"): # should be in the format of "pick_entropy_r9"
            # print(f"current i is {i}")
            # print("self.files len is ", len(self.files))
            mfp = self.fp_loader.build_mfp(int(dataset_files[i].split(".")[0]), current_dataset ,self.split)
        else:   
            mfp = torch.load(f"{dataset_dir}/{self.fp_suffix}/{dataset_files[i]}").float()  

        if self.parser_args['loss_func'] == "CE":
            num_class = self.parser_args['num_class']
            mfp = torch.where(mfp >= num_class, num_class-1, mfp).long()
            
        combined = (inputs, mfp, NMR_type_indicator)
        
        if self.parser_args['separate_classifier'] :
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
            combined = (inputs, mfp, NMR_type_indicator, mol_weight, torch.tensor(input_type))

        return combined
    
    def pad_and_stack_input(self, hsqc, c_tensor, h_tensor, mol_weight):
        c_tensor, h_tensor = c_tensor.view(-1, 1), h_tensor.view(-1, 1)
        c_tensor,h_tensor = F.pad(c_tensor, (0, 2), "constant", 0), F.pad(h_tensor, (1, 1), "constant", 0)
        inputs = [hsqc, c_tensor, h_tensor]
        NMR_type_indicator = [0]*len(hsqc)+[1]*len(c_tensor)+[2]*len(h_tensor)
        if mol_weight is not None and len(mol_weight) > 0:
            inputs.append(mol_weight)
            NMR_type_indicator.append(3)
            
        inputs = torch.vstack(inputs)               
        NMR_type_indicator = torch.tensor(NMR_type_indicator).long()
        return inputs, NMR_type_indicator
    
    def get_weight_of_samples_based_on_input_type(self):

        # ["all_inputs", "HSQC_H_NMR", "HSQC_C_NMR", "only_hsqc", "only_1d", "only_H_NMR",  "only_C_NMR"]
        path = f"{repo_path}/datasets/input_type_weights.pt"
        if os.path.exists(path):
            type_of_each_sample =  torch.load(path)
        else:
            type_of_each_sample = self.build_input_types_of_each(path)
        
        if self.parser_args['sampling_strategy'] == "fewer_HSQC":
            weight_map = {
                0: 1, # all_inputs
                1: 1, # HSQC_H_NMR
                2: 1, # HSQC_C_NMR
                3: 0.3, # only_hsqc
                4: 1, # only_1d
                5: 1, # only_H_NMR
                6: 1, # only_C_NMR       
            }
        elif self.parser_args['sampling_strategy'] == "match_probability":
            weight_map = {
                0: 1/0.04, # all_inputs
                1: 1/0.04, # HSQC_H_NMR
                2: 1/0.04, # HSQC_C_NMR
                3: 1/0.261, # only_hsqc
                4: 1/0.206, # only_1d
                5: 1/0.206, # only_H_NMR
                6: 1/0.206, # only_C_NMR       
            }    
        else:
            raise NotImplementedError(f"sampling strategy is not implemented yet: {self.parser_args['sampling_strategy']}")
        
        weights_of_sampling = torch.tensor([weight_map[i] for i in type_of_each_sample]).float()
        return weights_of_sampling
        

    def build_input_types_of_each(self, path):
        type_of_each_sample = [0]*self.__len__()
        for idx in range(self.__len__()):
            if idx >= len(self.files): # load 1D dataset
                i = idx - len(self.files)
                    # hsqc is empty tensor
                has_hsqc = False
                c_tensor, h_tensor = torch.load(f"{self.dir_1d}/oneD_NMR/{self.files_1d[i]}")
            else:
                i = idx
                c_tensor, h_tensor = (torch.tensor([]) , torch.tensor([])) 
                if os.path.exists(f"{self.dir}/oneD_NMR/"+ self.files[i]):
                    c_tensor, h_tensor = torch.load(f"{self.dir}/oneD_NMR/{self.files[i]}")  
                if os.path.exists(f"{self.dir}/HSQC/{self.files[i]}"):
                    has_hsqc = True
            has_c = len(c_tensor) > 0
            has_h = len(h_tensor) > 0
            input_type = 0
            if has_hsqc:
                input_type+=4
            if has_h:
                input_type+=2
            if has_c:
                input_type+=1
            input_type = 7-input_type    
        torch.save(type_of_each_sample, path)
        return type_of_each_sample
                
        
def get_solvent(solvent_name):
    match solvent_name:
        case "H2O": return torch.tensor([-9,-9,-9]).float()
        case "D2O": return torch.tensor([-10,-10,-10]).float()
        case "unknown": return torch.tensor([-11,-11,-11]).float()
        case "No_1D_NMR": return torch.tensor([]).float().view(-1,3) # empty tensor, will be skipped duing v-stack
        case _: raise Exception(f"unknown {solvent_name}")

# used as collate_fn in the dataloader
def pad(batch):
    items = tuple(zip(*batch))
    inputs = items[0]
    inputs = pad_sequence([v for v in inputs], batch_first=True) 
    
    fp = items[1]
    if type(fp[0][0]) is not str:
        fp = torch.stack(fp)
    if len(items) == 2:
        return (inputs, fp)
    
    NMR_type_indicator = pad_sequence([v for v in items[2]], batch_first=True)
    # mol_weight = torch.stack(items[3])
    # combined = (inputs, fp, NMR_type_indicator,mol_weight,  *items[4:])
    return (inputs, fp, NMR_type_indicator)
    # return combined
    

def normalize_hsqc(hsqc, style="minmax"):
    """
    Normalizes each column of the input HSQC to have zero mean and unit standard deviation.
    Parameters:
    hsqc (torch.Tensor): Input tensor of shape (n, 3).
    Returns:
    torch.Tensor: Normalized hsqc of shape (n, 3).
    """    
    
    assert(len(hsqc.shape)==2 and hsqc.shape[1]==3)
    input_signs = torch.sign(hsqc)
    copy_hsqc = hsqc.clone()
    '''cananical approach to normalize each field'''
    # Calculate the mean and standard deviation for each column
    
    
    '''normalize only peak intensities, and separate positive and negative peaks'''
    selected_values = hsqc[hsqc[:,2] > 0, 2]
    # do min_max normalization with in the range of 0.5 to 1.5
    if len(selected_values) > 1:
        min_pos = selected_values.min()
        max_pos = selected_values.max()
        if min_pos == min_pos:
            hsqc[hsqc[:,2]>0,2] = 1
        else:
            hsqc[hsqc[:,2]>0,2] = (selected_values - min_pos) / (max_pos - min_pos) + 0.5
    elif len(selected_values) == 1:
        hsqc[hsqc[:,2]>0,2] = 1
    
    # do min_max normalization with in the range of -0.5 to -1.5
    selected_values = hsqc[hsqc[:,2] < 0, 2]
    if len(selected_values) > 1:
        min_neg = selected_values.min()
        max_neg = selected_values.max()
        if min_neg == max_neg:
            hsqc[hsqc[:,2]<0,2] = -1
        else:
            hsqc[hsqc[:,2]<0,2] = (min_neg - selected_values ) / (max_neg - min_neg) - 0.5
    elif len(selected_values) == 1:
        hsqc[hsqc[:,2]<0,2] = -1
            
    # output_signs = torch.sign(hsqc)  
    # if (input_signs != output_signs).any():
    #     print("signs are changed")
    #     print(copy_hsqc)
    #     print(hsqc)
    #     exit(0)
    return hsqc
    

class FolderDataModule(pl.LightningDataModule):
    def __init__(self, dir, FP_choice, input_src, batch_size: int = 32, parser_args=None):
        super().__init__()
        self.batch_size = batch_size
        self.dir = dir
        self.FP_choice = FP_choice
        self.input_src = input_src
        self.collate_fn = pad
        self.parser_args = parser_args
    
    def setup(self, stage):
        if stage == "fit" or stage == "validate" or stage is None:
            self.train = FolderDataset(dir=self.dir, FP_choice=self.FP_choice, input_src = self.input_src, split="train", parser_args=self.parser_args)
            self.val = FolderDataset(dir=self.dir, FP_choice=self.FP_choice, input_src = self.input_src,split="val", parser_args=self.parser_args)
        if stage == "test":
            self.test = FolderDataset(dir=self.dir, FP_choice=self.FP_choice, input_src = self.input_src, split="test", parser_args=self.parser_args)
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