import logging
import pickle
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
        if parser_args['use_MS']:
            self.mass_spec = pickle.load(open(os.path.join(self.dir, "MW/index.pkl"), 'rb'))

        self.files = os.listdir(os.path.join(self.dir, "HYUN_FP"))
        logger = logging.getLogger("lightning")
        if dist.is_initialized():
            rank = dist.get_rank()
            if rank != 0:
                # For any process with rank other than 0, set logger level to WARNING or higher
                logger.setLevel(logging.WARNING)
        logger.info(f"[FolderDataset]: dir={dir},input_src={input_src},split={split},FP={FP_choice},normalize_hsqc={parser_args['normalize_hsqc']}")
        
    def __len__(self):
        # return 10000
        return len(self.files)

    def __getitem__(self, i):
        def file_exist(src, filename):
            return os.path.exists(os.path.join(self.dir, src, filename))
        
        # Load HSQC
        hsqc = torch.load(f"{self.dir}/HSQC/{self.files[i]}").type(torch.FloatTensor)
        if self.parser_args['disable_hsqc_peaks']:
            hsqc[:,2]=0
        if self.parser_args['normalize_hsqc']:
            hsqc = normalize_columns(hsqc)
        if self.parser_args['enable_hsqc_delimeter_only_2d']:
            assert (self.input_src == ["HSQC"])
            hsqc = torch.vstack([get_delimeter("HSQC_start"), hsqc, get_delimeter("HSQC_end")])   
        inputs = hsqc
        
        if "detailed_oneD_NMR" in self.input_src:
            c_tensor, h_tensor, solvent = torch.load(f"{self.dir}/detailed_oneD_NMR/{self.files[i]}") if file_exist("detailed_oneD_NMR", self.files[i]) \
                else (torch.tensor([]) , torch.tensor([]), "No_1D_NMR") # empty tensor, will be skipped duing v-stack
    
            c_tensor, h_tensor = c_tensor.view(-1, 1), h_tensor.view(-1, 1)
            c_tensor,h_tensor = F.pad(c_tensor, (0, 2), "constant", 0), F.pad(h_tensor, (1, 1), "constant", 0)
            inputs = torch.vstack([
                get_delimeter("HSQC_start"),  hsqc,     get_delimeter("HSQC_end"),
                get_delimeter("C_NMR_start"), c_tensor, get_delimeter("C_NMR_end"), 
                get_delimeter("H_NMR_start"), h_tensor, get_delimeter("H_NMR_end"),
                ])    
            if not self.parser_args['disable_solvent']: # add solvent info
                inputs = torch.vstack([inputs, get_delimeter("solvent_start"), get_solvent(solvent), get_delimeter("solvent_end")])
        elif "oneD_NMR" in self.input_src:
            c_tensor, h_tensor = torch.load(f"{self.dir}/oneD_NMR/{self.files[i]}") if file_exist("oneD_NMR", self.files[i]) else (torch.tensor([]) , torch.tensor([])) 
            c_tensor, h_tensor = c_tensor.view(-1, 1), h_tensor.view(-1, 1)
            c_tensor,h_tensor = F.pad(c_tensor, (0, 2), "constant", 0), F.pad(h_tensor, (0, 2), "constant", 0)
            inputs = torch.vstack([
                get_delimeter("HSQC_start"),  hsqc,     get_delimeter("HSQC_end"),
                get_delimeter("C_NMR_start"), c_tensor, get_delimeter("C_NMR_end"), 
                get_delimeter("H_NMR_start"), h_tensor, get_delimeter("H_NMR_end"),
                ])    
        if self.parser_args['use_MS']:
            mass_spec = self.mass_spec[int(self.files[i].split(".")[0])]
            mass_spec = torch.tensor([mass_spec,0,0]).float()
            inputs = torch.vstack([inputs, get_delimeter("ms_start"), mass_spec, get_delimeter("ms_end")])
               
        mfp = torch.load(f"{self.dir}/{self.fp_suffix}/{self.files[i]}")  
        combined = (inputs, mfp.type(torch.FloatTensor))
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

def pad(batch):
    items = tuple(zip(*batch))
    fp = items[-1]
    inputs = items[:-1]
    inputs_2 = [pad_sequence([v for v in input], batch_first=True) for input in inputs]
    combined = (*inputs_2, torch.stack(fp))
    return combined

def normalize_columns(hsqc):
    """
    Normalizes each column of the input HSQC to have zero mean and unit standard deviation.
    Parameters:
    hsqc (torch.Tensor): Input tensor of shape (n, 3).
    Returns:
    torch.Tensor: Normalized hsqc of shape (n, 3).
    """    
    
    assert(len(hsqc.shape)==2 and hsqc.shape[1]==3)
    # Calculate the mean and standard deviation for each column
    mean = hsqc.mean(dim=0, keepdim=True)
    std = hsqc.std(dim=0, keepdim=True, unbiased=False)
    
    # Normalize each column
    normalized_hsqc = (hsqc - mean) / std
    
    return normalized_hsqc

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