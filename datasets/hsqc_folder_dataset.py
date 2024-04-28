import logging
import pickle, random
import torch, os, pytorch_lightning as pl, glob
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F


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
        self.dir = os.path.join(dir, split)
        self.split = split
        self.fp_suffix = FP_choice
        self.input_src = input_src
        self.parser_args = parser_args

        print(self.dir)
        assert(os.path.exists(self.dir))
        assert(split in ["train", "val", "test"])
        for src in input_src:
            assert os.path.exists(os.path.join(self.dir, src)),"{} does not exist".format(os.path.join(self.dir, src))
        if parser_args['use_MW']:
            self.mol_weight_2d = pickle.load(open(os.path.join(self.dir, "MW/index.pkl"), 'rb'))
        self.files = os.listdir(os.path.join(self.dir, "HYUN_FP"))
        
        if "oneD_NMR" in self.input_src: # load 1D dataset as well 
            self.dir_1d = f"/workspace/OneD_Only_Dataset/{split}"
            if parser_args['combine_oneD_only_dataset']:
                self.mol_weight_1d = pickle.load(open(os.path.join(self.dir_1d, "MW/index.pkl"), 'rb'))
            self.files_1d = os.listdir(os.path.join(self.dir_1d, "oneD_NMR/"))
            
        # path_to_load_full_info_indices = f"/root/MorganFP_prediction/reproduce_previous_works/smart4.5/datasets/{split}_indices_of_full_info_NMRs.pkl"
        # self.files = pickle.load(open(path_to_load_full_info_indices, "rb"))
        
        logger = logging.getLogger("lightning")
        if dist.is_initialized():
            rank = dist.get_rank()
            if rank != 0:
                # For any process with rank other than 0, set logger level to WARNING or higher
                logger.setLevel(logging.WARNING)
        logger.info(f"[FolderDataset]: dir={dir},input_src={input_src},split={split},FP={FP_choice},normalize_hsqc={parser_args['normalize_hsqc']}")
        
    def __len__(self):
        # return 100
        length = len(self.files)
        if self.parser_args['combine_oneD_only_dataset']:
            length += len(self.files_1d)
        return length
        


    def __getitem__(self, idx):
        
        if idx >= len(self.files): # load 1D dataset
            i = idx - len(self.files)
            # hsqc is empty tensor
            hsqc = torch.empty(0,3)
            c_tensor, h_tensor = torch.load(f"{self.dir_1d}/oneD_NMR/{self.files_1d[i]}")
            if self.parser_args['optional_inputs'] and len(c_tensor) > 0 and len(h_tensor) > 0:
                random_num = random.random()
                if random_num <= 1/3:
                    c_tensor = torch.tensor([]) 
                elif random_num <= 2/3:
                    h_tensor = torch.tensor([])
            c_tensor, h_tensor = c_tensor.view(-1, 1), h_tensor.view(-1, 1)
            c_tensor,h_tensor = F.pad(c_tensor, (0, 2), "constant", 0), F.pad(h_tensor, (0, 2), "constant", 0)

            
            inputs = torch.vstack([
                    get_delimeter("HSQC_start"),  hsqc,     get_delimeter("HSQC_end"),
                    get_delimeter("C_NMR_start"), c_tensor, get_delimeter("C_NMR_end"), 
                    get_delimeter("H_NMR_start"), h_tensor, get_delimeter("H_NMR_end"),
                    ])    
            
            
            
        else :
            ### BEGINNING 2D dataset case
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
                
            
                if self.parser_args['use_peak_values']:
                    hsqc = normalize_hsqc(hsqc)
                if self.parser_args['enable_hsqc_delimeter_only_2d']:
                    assert (self.input_src == ["HSQC"])
                    hsqc = torch.vstack([get_delimeter("HSQC_start"), hsqc, get_delimeter("HSQC_end")])   
                inputs = hsqc
            
            if "oneD_NMR" in self.input_src:
                if file_exist("oneD_NMR", self.files[i]):
                    c_tensor, h_tensor = torch.load(f"{self.dir}/oneD_NMR/{self.files[i]}")  
                    # randomly drop 1D and 2D NMRs if needed
                    if self.parser_args['optional_inputs']:
                        # drop 2D:
                        if random.random() <= 1/2: 
                            hsqc =  torch.empty(0,3)
                        if len(c_tensor)>0 and len(h_tensor)>0:
                            # it is fine to drop one of the oneD NMRs
                            random_num_for_dropping =  random.random()
                            
                            if random_num_for_dropping <= 1/3:
                                c_tensor = torch.tensor([])
                            elif random_num_for_dropping <= 2/3:
                                h_tensor = torch.tensor([])
                            # else: keep both
                                
                    
                        assert (len(hsqc) > 0 or len(c_tensor) > 0 or len(h_tensor) > 0), "all NMRs are dropped"
                            
                else:
                    c_tensor, h_tensor = (torch.tensor([]) , torch.tensor([])) 
                    # then we should not drop 2D info anyawy
                    
                c_tensor, h_tensor = c_tensor.view(-1, 1), h_tensor.view(-1, 1)
                c_tensor,h_tensor = F.pad(c_tensor, (0, 2), "constant", 0), F.pad(h_tensor, (0, 2), "constant", 0)
                inputs = torch.vstack([
                    get_delimeter("HSQC_start"),  hsqc,     get_delimeter("HSQC_end"),
                    get_delimeter("C_NMR_start"), c_tensor, get_delimeter("C_NMR_end"), 
                    get_delimeter("H_NMR_start"), h_tensor, get_delimeter("H_NMR_end"),
                    ])    
                
            ### ENDING 2D dataset case
            
            
        # loading MW and MFP in different datasets 
        if idx >= len(self.files): # load 1D dataset    
            mol_weight_dict = self.mol_weight_1d
            dataset_files = self.files_1d
            dataset_dir = self.dir_1d
        else:
            mol_weight_dict = self.mol_weight_2d
            dataset_files = self.files
            dataset_dir = self.dir
            
        if self.parser_args['use_MW']:
            mol_weight = mol_weight_dict[int(dataset_files[i].split(".")[0])]
            mol_weight = torch.tensor([mol_weight,0,0]).float()
            inputs = torch.vstack([inputs, get_delimeter("ms_start"), mol_weight, get_delimeter("ms_end")])
            
        mfp = torch.load(f"{dataset_dir}/{self.fp_suffix}/{dataset_files[i]}").float()  
        if self.parser_args['loss_func'] == "CE":
            num_class = self.parser_args['num_class']
            mfp = torch.where(mfp >= num_class, num_class-1, mfp).long()
        
            
            
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
   


def get_delimeter(delimeter_name):
    match delimeter_name:
        case "HSQC_start":
            return torch.tensor([-1,-1,-1]).float()
        case "HSQC_end":
            return torch.tensor([-2,-2,-2]).float()
        case "C_NMR_start":
            return torch.tensor([-3,-3,-3]).float()
        case "C_NMR_end":
            return torch.tensor([-4,-4,-4]).float()
        case "H_NMR_start":
            return torch.tensor([-5,-5,-5]).float()
        case "H_NMR_end":
            return torch.tensor([-6,-6,-6]).float()
        case "solvent_start":
            return torch.tensor([-7,-7,-7]).float()
        case "solvent_end":
            return torch.tensor([-8,-8,-8]).float()
        case "ms_start":
            return torch.tensor([-12,-12,-12]).float()
        case "ms_end":
            return torch.tensor([-13,-13,-13]).float()
        case _:
            raise Exception(f"unknown {delimeter_name}")
                    
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
    if len(items) == 2: #inputs, mfp,
        fp = items[-1]
        inputs = items[0]
        inputs_2 = pad_sequence([v for v in inputs], batch_first=True) 
        # print(fp)h
        if type(fp[0][0]) is str:
            # print("i am tuple")
            # print(fp)
            combined = (inputs_2, fp) # actually, here fp is (smiles, name), used during prediction stage
        else:
            combined = (inputs_2, torch.stack(fp))
    if len(items) == 3: #inputs, mfp, input_type(optional input)
        input_type = items[-1]
        fp = items[-2]
        inputs = items[0]
        inputs_2 = pad_sequence([v for v in inputs], batch_first=True) 
        combined = (inputs_2, torch.stack(fp), torch.tensor(input_type))
    return combined
    

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