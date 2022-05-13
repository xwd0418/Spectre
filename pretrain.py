import torch, pytorch_lightning as pl
import pytorch_lightning.callbacks as cb
# from argparse import ArgumentParser
from datasets.ms_dataset import MsDataModule
from pytorch_lightning.loggers import TensorBoardLogger
from models.encoder_only import EncoderOnly

def main():
    # parser = ArgumentParser(add_help=True)
    # parser.add_argument("--model", type=str, default="parg1")
    out_path = "/workspace/volume/tensorboard"

    model = EncoderOnly()
    data_module = MsDataModule(batch_size=128)

    tbl = TensorBoardLogger(save_dir=out_path, name="lightning_logs", version="pretrain")
    checkpoint_callback = cb.ModelCheckpoint(monitor="val/loss", mode="min", save_last=True)
    trainer = pl.Trainer(max_epochs=60, gpus=1, logger=tbl, callbacks=[checkpoint_callback])
    trainer.fit(model, data_module)
    trainer.test(model, data_module)

if __name__ == '__main__':
    main()
