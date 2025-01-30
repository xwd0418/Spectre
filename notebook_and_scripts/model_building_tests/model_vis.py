# 09/10/22 the purpose of this file is to experiment with visualizing a model in tensoboard with pytorch lightning
import torch, torch.nn as nn, pytorch_lightning as pl
import pytorch_lightning.callbacks as cb
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Callback, LightningModule, Trainer


class MyCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training is starting")

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule):
        print("Fitting is ending")
        z = torch.normal(mean=0.0, std=1.0, size=(5, 1), device=pl_module.device)
        with torch.no_grad():
            pl_module.eval()
            out = pl_module(z)
            pl_module.train()
        trainer.logger.experiment.add_graph(pl_module, z)

class DummyModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.e = 0
        self.l = nn.Linear(1, 3)
        self.loss = nn.MSELoss()
        self.d = [5, 6, 7, 8]
        pass
    def forward(self, data):
        return self.l(data)
    def fn(self, e):
        return self.d[e]
    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss(self.forward(x), y)
        return loss
    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss(self.forward(x), y)
        return {"loss": loss}
    def validation_epoch_end(self, validation_step_outputs):
        self.log('val/my_loss', self.fn(self.e))
        self.e += 1
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

class DS(Dataset):
    def __init__(self):
        super().__init__()
    def __len__(self):
        return 10
    def __getitem__(self, i):
        return torch.tensor([i]).float(), torch.tensor([i, i, i]).float()
class DM(pl.LightningDataModule):
    def setup(self, stage):
        if stage == "fit" or stage is None:
            self.train = DS()
            self.val = DS()
        if stage == "test":
            self.test = DS()
        if stage == "predict":
            raise NotImplementedError("Predict setup not implemented")
    def train_dataloader(self):
        return DataLoader(self.train, shuffle=True, batch_size=64, num_workers=4)
    def val_dataloader(self):
        return DataLoader(self.val, batch_size=64, num_workers=4)
    def test_dataloader(self):
        return DataLoader(self.test, batch_size=64, num_workers=4)

def main():
    model, dm = DummyModule(), DM()
    tbl = TensorBoardLogger(save_dir="/workspace/Spectre/ignore", name="path1", version="v1")
    cb = EarlyStopping(monitor="val/my_loss", mode="min", verbose=True, patience=1)
    cb2 = MyCallback()
    trainer = pl.Trainer(max_epochs=5, gpus=0, logger = tbl, callbacks=[cb, cb2])
    trainer.fit(model, dm)


if __name__ == '__main__':
    main()