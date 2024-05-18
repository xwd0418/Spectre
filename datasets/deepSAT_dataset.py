import logging
import pickle, random, json
import torch, os, pytorch_lightning as pl, glob
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from datasets.dataset_utils import specific_radius_mfp_loader



class DeepSATDataset(Dataset):
    '''
        
    '''
    def __init__(self, dir, split="train" , parser_args=None):
        self.dir = os.path.join(dir, split)
        self.split = split
        self.parser_args = parser_args
        logger = logging.getLogger("lightning")

        assert(os.path.exists(self.dir))
        assert(split in ["train", "val", "test"])

        self.files = os.listdir(os.path.join(self.dir, "HYUN_FP"))
        self.files.sort() # sorted because we need to find correct weight mappings 
    
        self.mol_weight_2d = pickle.load(open(os.path.join(self.dir, "MW/index.pkl"), 'rb'))
        self.idx_to_superclass_and_gly = pickle.load(open(f'/root/MorganFP_prediction/reproduce_previous_works/reproducing_deepsat/NP_classify_scrape/NP_class_{split}.pkl', 'rb'))
        self.cls_name_to_label = json.load(open(f'/root/MorganFP_prediction/reproduce_previous_works/reproducing_deepsat/index_v1.json', 'r'))['Superclass']
        self.cls_name_to_label['Sphingolipids']=77
        
        if dist.is_initialized():
            rank = dist.get_rank()
            if rank != 0:
                # For any process with rank other than 0, set logger level to WARNING or higher
                logger.setLevel(logging.WARNING)
        logger.info(f"[DeepSAT rDataset]: dir={dir},split={split},FP=HYUN_FP']")
        
        
        
    def __len__(self):
        # return 500
        length = len(self.files)
        return length
        


    def __getitem__(self, idx):
        
        # inputs = torch.load(f"{self.dir}/HSQC_images_1channel/{self.files[idx]}")
        inputs = torch.load(f"{self.dir}/HSQC_images_2channel/{self.files[idx]}")
            
        
        mol_weight = self.mol_weight_2d[int(self.files[idx].split(".")[0])]
        super_class, gly = self.idx_to_superclass_and_gly[int(self.files[idx].split(".")[0])]
        if len(super_class)!=1:
            super_class_label = -1
        else:
            super_class_label = self.cls_name_to_label[super_class[0]] 
        mfp = torch.load(f"{self.dir}/HYUN_FP/{self.files[idx]}").float()  
            
            
        combined = inputs, {
            "fp": mfp,
            "mw": torch.tensor([mol_weight]).float(),
            "np_cls": super_class_label,
            "gly": int(gly)
        }
        return combined
    

class DeepSATDataModule(pl.LightningDataModule):
    def __init__(self, dir, batch_size: int = 32, parser_args=None):
        super().__init__()
        self.batch_size = batch_size
        self.dir = dir
        self.parser_args = parser_args
    
    def setup(self, stage):
        if stage == "fit" or stage == "validate" or stage is None:
            self.train = DeepSATDataset(dir=self.dir,   split="train", parser_args=self.parser_args)
            self.val = DeepSATDataset(dir=self.dir,  split="val", parser_args=self.parser_args)
        if stage == "test":
            self.test = DeepSATDataset(dir=self.dir,   split="test", parser_args=self.parser_args)
        if stage == "predict":
            raise NotImplementedError("Predict setup not implemented")

    def train_dataloader(self):
            
        return DataLoader(self.train, shuffle=True, batch_size=self.batch_size, 
                          num_workers=self.parser_args['num_workers'], pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=1,  
                          num_workers=self.parser_args['num_workers'], pin_memory=True, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=1,  
                          num_workers=self.parser_args['num_workers'], pin_memory=True, persistent_workers=True)
    