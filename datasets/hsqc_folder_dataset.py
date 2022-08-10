import logging
import torch, os, pytorch_lightning as pl, glob
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

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
    def __init__(self, dir="/workspace/smart4.5/tempdata/hyun_fp_data/hsqc_ms_pairs", split="train", input_src=["HSQC"], do_hyun_fp=True):
        self.dir = os.path.join(dir, split)
        self.split = split
        self.fp_suffix = "HYUN_FP" if do_hyun_fp else "FP"
        self.input_src = input_src

        assert(os.path.exists(self.dir))
        assert(split in ["train", "val", "test"])
        for src in input_src:
            assert(os.path.exists(os.path.join(self.dir, src)))

        self.files = os.listdir(os.path.join(self.dir, "FP"))
        logger = logging.getLogger("lightning")
        logger.info(f"[FolderDataset]: dir={dir},input_src={input_src},split={split},hyunfp={do_hyun_fp}")
    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        inputs = [torch.load(f"{self.dir}/{src}/{self.files[i]}").type(torch.FloatTensor) for src in self.input_src]
        mfp = torch.load(f"{self.dir}/{self.fp_suffix}/{self.files[i]}")
        combined = (*inputs, mfp.type(torch.FloatTensor))
        return combined

def pad(batch):
    items = tuple(zip(*batch))
    fp = items[-1]
    inputs = items[:-1]
    inputs_2 = [pad_sequence([v for v in input], batch_first=True) for input in inputs]
    combined = (*inputs_2, torch.stack(fp))
    return combined

class FolderDataModule(pl.LightningDataModule):
    def __init__(self, dir, do_hyun_fp, input_src, batch_size: int = 32):
        super().__init__()
        self.batch_size = batch_size
        self.dir = dir
        self.do_hyun_fp = do_hyun_fp
        self.input_src = input_src
        self.collate_fn = pad
    
    def setup(self, stage):
        if stage == "fit" or stage is None:
            self.train = FolderDataset(dir=self.dir, do_hyun_fp=self.do_hyun_fp, input_src = self.input_src, split="train")
            self.val = FolderDataset(dir=self.dir, do_hyun_fp=self.do_hyun_fp, input_src = self.input_src,split="val")
        if stage == "test":
            self.test = FolderDataset(dir=self.dir, do_hyun_fp=self.do_hyun_fp, input_src = self.input_src, split="test")
        if stage == "predict":
            raise NotImplementedError("Predict setup not implemented")

    def train_dataloader(self):
        return DataLoader(self.train, shuffle=True, batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=4)