import torch, os, pytorch_lightning as pl, glob
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

class HSQCFolderDataset(Dataset):
    def __init__(self, dir="/workspace/smart4.5/tempdata/hyun_fp_data/hsqc_ms_pairs", split="train", do_hyun_fp=True):
        self.dir = os.path.join(dir, split)
        self.split = split
        self.fp_suffix = "HYUN_FP" if do_hyun_fp else "FP"

        assert(os.path.exists(self.dir))
        assert(split in ["train", "val", "test"])

        self.files = os.listdir(os.path.join(self.dir, "FP"))
    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        hsqc = torch.load(f"{self.dir}/HSQC/{self.files[i]}")
        mfp =  torch.load(f"{self.dir}/{self.fp_suffix}/{self.files[i]}")
        return hsqc.type(torch.FloatTensor), mfp.type(torch.FloatTensor)

def pad(batch):
    hsqc, fp = zip(*batch)
    hsqc = pad_sequence([torch.tensor(v, dtype=torch.float) for v in hsqc], batch_first=True)
    return hsqc, torch.stack(fp)

class HsqcDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32):
        super().__init__()
        self.batch_size = batch_size
        self.collate_fn = pad
    
    def setup(self, stage):
        if stage == "fit" or stage is None:
            self.train = HSQCFolderDataset(split="train")
            self.val = HSQCFolderDataset(split="val")
        if stage == "test":
            self.test = HSQCFolderDataset(split="test")
        if stage == "predict":
            raise NotImplementedError("Predict setup not implemented")

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=4)