import torch, pytorch_lightning as pl
import pytorch_lightning.callbacks as cb
from encoder import MassSpecEncoder
from argparse import ArgumentParser
from datasets.pair_dataset import PairDataModule
from pytorch_lightning.loggers import TensorBoardLogger
from models.pair_net import PairNet
import logging, os, sys

def main():
    parser = ArgumentParser(add_help=True)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--lr", type=float, default=1e-5)
    args = vars(parser.parse_args())

    lr, epochs = args["lr"], args["epochs"]

    out_path = "/data/smart4.0"
    path1, path2 = "lightning_logs", "poggers2"

    model = PairNet(n_layers = 4, n_head = 4)
    data_module = PairDataModule(batch_size=32)

    # === Init Logger ===
    logger = logging.getLogger("lightning")
    logger.setLevel(logging.DEBUG)
    file_path = os.path.join(out_path, path1, path2, "logs.txt")
    os.makedirs(os.path.join(out_path, path1, path2), exist_ok=True)
    with open(file_path, 'w') as fp: # touch
        pass
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler(file_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    # === End Init Logger === 

    logger.info(f"Basic Params: [LR: {lr}, EPOCHS: {epochs}]")

    tbl = TensorBoardLogger(save_dir=out_path, name=path1, version=path2)
    checkpoint_callback = cb.ModelCheckpoint(monitor="val/mean_ce_loss", mode="min", save_last=True)
    trainer = pl.Trainer(max_epochs=160, gpus=1, logger=tbl, callbacks=[checkpoint_callback])
    trainer.fit(model, data_module)

if __name__ == '__main__':
    main()
