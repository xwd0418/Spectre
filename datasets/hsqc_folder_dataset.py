import logging
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
    def __init__(self, dir, split="train", input_src=["HSQC"], do_hyun_fp=True, parser_args=None):
        self.dir = os.path.join(dir, split)
        self.split = split
        self.fp_suffix = "HYUN_FP" if do_hyun_fp else "R2-6144FP"
        self.input_src = input_src
        self.parser_args = parser_args

        print(self.dir)
        assert(os.path.exists(self.dir))
        assert(split in ["train", "val", "test"])
        for src in input_src:
            assert(os.path.exists(os.path.join(self.dir, src)))

        self.files = os.listdir(os.path.join(self.dir, "HYUN_FP"))
        logger = logging.getLogger("lightning")
        if dist.is_initialized():
            rank = dist.get_rank()
            if rank != 0:
                # For any process with rank other than 0, set logger level to WARNING or higher
                logger.setLevel(logging.WARNING)
        logger.info(f"[FolderDataset]: dir={dir},input_src={input_src},split={split},hyunfp={do_hyun_fp},normalize_hsqc={parser_args['normalize_hsqc']}")
        
    def __len__(self):
        # return 10000
        return len(self.files)

    def __getitem__(self, i):
        def file_exist(src, filename):
            return os.path.exists(os.path.join(self.dir, src, filename))
        
        inputs = [torch.load(f"{self.dir}/HSQC/{self.files[i]}").type(torch.FloatTensor) ]
        if "detailed_oneD_NMR" in self.input_src:
            c_tensor, h_tensor, solvent = torch.load(f"{self.dir}/detailed_oneD_NMR/{self.files[i]}") if file_exist("detailed_oneD_NMR", self.files[i]) else (torch.tensor([]) , torch.tensor([]), "No_1D_NMR")  # C, H, solvent
            c_tensor, h_tensor = c_tensor.view(-1, 1), h_tensor.view(-1, 1)
            c_tensor,h_tensor = F.pad(c_tensor, (0, 2), "constant", 0), F.pad(h_tensor, (1, 1), "constant", 0)
            inputs+=[c_tensor, h_tensor, solvent]
        mfp = torch.load(f"{self.dir}/{self.fp_suffix}/{self.files[i]}")
        combined = (*inputs, mfp.type(torch.FloatTensor))
        return combined
    
        # inputs=[]
        if "HSQC" in self.input_src:
            hsqc = torch.load(f"{self.dir}/HSQC/{self.files[i]}").type(torch.FloatTensor)
            if self.parser_args['normalize_hsqc']:
                hsqc = normalize_columns(hsqc)
            inputs = hsqc
        if "detailed_oneD_NMR" in self.input_src:
            inputs = torch.vstack([get_delimeter("HSQC_start"),hsqc,get_delimeter("HSQC_end")])
            c_tensor, h_tensor = torch.load(f"{self.dir}/detailed_oneD_NMR/{self.files[i]}") if file_exist("detailed_oneD_NMR", self.files[i]) else (torch.tensor([]) , torch.tensor([])) 
            c_tensor, h_tensor = c_tensor.view(-1, 1), h_tensor.view(-1, 1)
            c_tensor,h_tensor = F.pad(c_tensor, (0, 2), "constant", 0), F.pad(h_tensor, (0, 2), "constant", 0)
            inputs = torch.vstack([inputs, get_delimeter("C_NMR_start"),c_tensor,get_delimeter("C_NMR_end"), get_delimeter("H_NMR_start"),h_tensor,get_delimeter("H_NMR_end"), ])    
        mfp = torch.load(f"{self.dir}/{self.fp_suffix}/{self.files[i]}")
        combined = (inputs, mfp.type(torch.FloatTensor))
        return combined

# def get_delimeter(delimeter_name):
#     match delimeter_name:
#         case "HSQC_start":
#             return torch.tensor([-1,-1,-1])
#         case "HSQC_end":
#             return torch.tensor([-2,-2,-2])
#         case "C_NMR_start":
#             return torch.tensor([-3,-3,-3])
#         case "C_NMR_end":
#             return torch.tensor([-4,-4,-4])
#         case "H_NMR_start":
#             return torch.tensor([-5,-5,-5])
#         case "H_NMR_end":
#             return torch.tensor([-6,-6,-6])
#         case _:
#             raise Exception(f"unknown {delimeter_name}")
        
solvent_index_lookup = {
                'D2O': 0, 'H2O': 1, 'unknown': 2, "No_1D_NMR":-1
            }            

def pad(batch):
    items = tuple(zip(*batch))
    fp = items[-1]
    if len(items)==2: # only HSQC
        inputs = items[:-1]
        inputs_2 = [pad_sequence([v for v in input], batch_first=True) for input in inputs]
        combined = (*inputs_2, torch.stack(fp))
    else: # hsqc+H+C+solvent
        H = items[2]
        # print("inspecting H tensor during padding:", type(H), H)
        inputs = items[:-2]
        solvent_indices = [solvent_index_lookup[solvent] for solvent in items[-2]]
        inputs_2 = [pad_sequence([v for v in input], batch_first=True) for input in inputs]
        combined = (*inputs_2, solvent_indices, torch.stack(fp))
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
    def __init__(self, dir, do_hyun_fp, input_src, batch_size: int = 32, parser_args=None):
        super().__init__()
        self.batch_size = batch_size
        self.dir = dir
        self.do_hyun_fp = do_hyun_fp
        self.input_src = input_src
        self.collate_fn = pad
        self.parser_args = parser_args
    
    def setup(self, stage):
        if stage == "fit" or stage == "validate" or stage is None:
            self.train = FolderDataset(dir=self.dir, do_hyun_fp=self.do_hyun_fp, input_src = self.input_src, split="train", parser_args=self.parser_args)
            self.val = FolderDataset(dir=self.dir, do_hyun_fp=self.do_hyun_fp, input_src = self.input_src,split="val", parser_args=self.parser_args)
        if stage == "test":
            self.test = FolderDataset(dir=self.dir, do_hyun_fp=self.do_hyun_fp, input_src = self.input_src, split="test", parser_args=self.parser_args)
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