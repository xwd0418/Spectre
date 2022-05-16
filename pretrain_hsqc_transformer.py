import pytorch_lightning as pl
import pytorch_lightning.callbacks as cb
from argparse import ArgumentParser
from datasets.hsqc_dataset_new import HsqcDataModule
from pytorch_lightning.loggers import TensorBoardLogger
from models.hsqc_transformer import HsqcTransformer
import logging, os, sys

def main():
    parser = ArgumentParser(add_help=True)
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--lr", type=float, default=1e-5)
    args = vars(parser.parse_args())

    lr, epochs = args["lr"], args["epochs"]

    out_path = "/workspace/volume/tensorboard"
    path1, path2 = "lightning_logs", "HsqcTransformer"

    model = HsqcTransformer(lr=lr,
                            dim_model=1024,
                            dim_coords=(342,341,341),
                            n_heads=8,
                            n_layers=3,
                            wavelength_bounds=[(None, 150),(None, 10),(None,20000)])
    data_module = HsqcDataModule(batch_size=128)

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
    checkpoint_callback = cb.ModelCheckpoint(monitor="val/loss", mode="min", save_last=True)
    trainer = pl.Trainer(max_epochs=epochs, gpus=1, logger=tbl, callbacks=[checkpoint_callback])
    trainer.fit(model, data_module)
    trainer.test(model, data_module)

if __name__ == '__main__':
    main()
