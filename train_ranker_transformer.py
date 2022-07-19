import torch, pytorch_lightning as pl
import pytorch_lightning.callbacks as cb
from argparse import ArgumentParser

from pytorch_lightning.loggers import TensorBoardLogger

from models.ranked_transformer import HsqcRankedTransformer
from datasets.hsqc_folder_dataset import HsqcDataModule
import logging, os, sys
from datetime import datetime
from pytz import timezone
def get_curr_time():
    pst = timezone("PST8PDT")
    california_time = datetime.now(pst)
    return california_time.strftime("%m_%d_%Y_%H:%M")


def main():
    # dependencies: hyun_fp_data, hyun_pair_ranking_set_07_22
    parser = ArgumentParser(add_help=True)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--expname", type=str, default=f"experiment_{get_curr_time()}")
    args = vars(parser.parse_args())

    lr, epochs = args["lr"], args["epochs"]

    out_path = "/data/smart4.5"
    path1, path2 = "lightning_logs", f"hsqc_only_ranked_4layers_4heads_{get_curr_time()}"

    model = HsqcRankedTransformer(lr=lr, n_layers=4, n_heads=4, dim_coords=(56, 56, 16))
    data_module = HsqcDataModule(batch_size=64)

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
    trainer = pl.Trainer(max_epochs=epochs, gpus=1, logger=tbl, callbacks=[checkpoint_callback])
    trainer.fit(model, data_module)

if __name__ == '__main__':
    main()
