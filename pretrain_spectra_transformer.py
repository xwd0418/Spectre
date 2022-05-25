import pytorch_lightning as pl
import pytorch_lightning.callbacks as cb
from argparse import ArgumentParser
from datasets.ms_dataset import MsDataModule
from pytorch_lightning.loggers import TensorBoardLogger
from models.spectra_transformer import SpectraTransformer
import logging, os, sys

def main():
    parser = ArgumentParser(add_help=True)
    parser.add_argument("--logdir", type=str, default="/workspace/volume/tensorboard")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--dim_model", type=int, default=1024)
    parser.add_argument("--dim_mz", type=int, default=512)
    parser.add_argument("--dim_intensity", type=int, default=512)
    parser.add_argument("--dim_feedforward", type=int, default=1024)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=.1)
    parser.add_argument("--mz_max_wave", type=int, default=2000)
    parser.add_argument("--intensity_max_wave", type=int, default=1)
    args = vars(parser.parse_args())

    lr, epochs = args["lr"], args["epochs"]
    dim_coords = (args["dim_mz"], args["dim_intensity"])

    out_path = args["logdir"]
    path1 = "SpectraTransformer"
    os.makedirs(os.path.join(out_path, path1), exist_ok=True)
    path2 = str(len(os.listdir(os.path.join(out_path, path1))))
    os.makedirs(os.path.join(out_path, path1, path2), exist_ok=True)

    model = SpectraTransformer(lr=lr,
                               dim_model=args["dim_model"], 
                               dim_coords=dim_coords,
                               dim_feedforward=args["dim_feedforward"],
                               n_heads=args["n_heads"],
                               n_layers=args["n_layers"],
                               dropout=args["dropout"],
                               wavelength_bounds=[(None, args["mz_max_wave"]),(None, args["intensity_max_wave"])])
    data_module = MsDataModule(batch_size=128)

    # === Init Logger ===
    logger = logging.getLogger("lightning")
    logger.setLevel(logging.DEBUG)
    file_path = os.path.join(out_path, path1, path2, "logs.txt")
    with open(file_path, 'w') as fp: # touch
        pass
    formatter = logging.Formatter('%(asctime)s - %(levelname)s\n%(message)s')
    fh = logging.FileHandler(file_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    # === End Init Logger === 

    logger.info(f"EPOCHS: {epochs}")

    tbl = TensorBoardLogger(save_dir=out_path, name=path1, version=path2)
    checkpoint_cb = cb.ModelCheckpoint(monitor="val/loss", mode="min", save_last=True)
    earlystop_cb = cb.EarlyStopping(monitor="val/loss", mode="min", patience=5)
    trainer = pl.Trainer(max_epochs=epochs, gpus=1, logger=tbl, callbacks=[checkpoint_cb, earlystop_cb])
    trainer.fit(model, data_module)
    results = trainer.test(model, data_module)
    logger.info(f'Test Cosine Similarity: {results[0]["test/cosine_sim"]}\nTest F1 Score: {results[0]["test/f1"]}\nTest Loss: {results[0]["test/loss"]}')

if __name__ == '__main__':
    main()
