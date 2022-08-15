import torch, pytorch_lightning as pl
import pytorch_lightning.callbacks as cb
from argparse import ArgumentParser
from functools import reduce
from pytorch_lightning.loggers import TensorBoardLogger

from models.ranked_transformer import HsqcRankedTransformer
from models.ranked_double_transformer import DoubleTransformer
from datasets.hsqc_folder_dataset import FolderDataModule
import logging, os, sys
from datetime import datetime
from pytz import timezone
def get_curr_time():
    pst = timezone("PST8PDT")
    california_time = datetime.now(pst)
    return california_time.strftime("%m_%d_%Y_%H:%M")

def data_mux(parser, model_type, data_src, do_hyun_fp, batch_size):
    my_dir = f"/workspace/smart4.5/tempdata/hyun_fp_data/{data_src}"
    if model_type == "double_transformer":
        return FolderDataModule(dir=my_dir, do_hyun_fp=do_hyun_fp, input_src=["HSQC", "MS"], batch_size=batch_size)
    elif model_type == "hsqc_transformer":
        return FolderDataModule(dir=my_dir, do_hyun_fp=do_hyun_fp, input_src=["HSQC"], batch_size=batch_size)
    elif model_type == "ms_transformer":
        return FolderDataModule(dir=my_dir, do_hyun_fp=do_hyun_fp, input_src=["MS"], batch_size=batch_size)
    raise(f"No datamodule for model type {model_type}.")


def model_mux(parser, model_type):
    eliminate = ["modelname", "debug", "expname", "foldername", "datasrc"]
    if model_type == "hsqc_transformer" or model_type == "ms_transformer":
        HsqcRankedTransformer.add_model_specific_args(parser)
        kwargs = vars(parser.parse_args())
        for v in eliminate:
            del kwargs[v]
        return HsqcRankedTransformer(**kwargs) 
    elif model_type == "double_transformer":
        DoubleTransformer.add_model_specific_args(parser)
        kwargs = vars(parser.parse_args())
        for v in eliminate:
            del kwargs[v]
        return DoubleTransformer(**kwargs) 
    raise(f"No model for model type {model_type}.")

def init_logger(out_path, path1, path2):
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
    return logger

def main():
    # dependencies: hyun_fp_data, hyun_pair_ranking_set_07_22
    parser = ArgumentParser(add_help=True)

    parser.add_argument("modelname", type=str)
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--expname", type=str, default=f"experiment")
    parser.add_argument("--foldername", type=str, default=f"lightning_logs")
    parser.add_argument("--datasrc", type=str, default=f"hsqc_ms_pairs")
    parser.add_argument("--bs", type=int, default=64)
    exclude = ["modelname", "debug", "expname", "foldername", "datasrc"] + ["hsqc_weights", "ms_weights"]
    args, _ = parser.parse_known_args()
    args = vars(args)

    model = model_mux(parser, args["modelname"])
    data_module = data_mux(parser, args["modelname"], args["datasrc"], True, args["bs"])
    print(parser.parse_args())

    args_with_model, _ = parser.parse_known_args()
    li_args = list(args_with_model.items())
    hyparam_string = "_".join([f"{hyparam}={val}"for hyparam, val in li_args if hyparam not in exclude])
    out_path = "/data/smart4.5"
    path1, path2 = args["foldername"], args["expname"] if args["debug"] else f"{args['expname']}_{get_curr_time()}_{hyparam_string}"

    logger = init_logger(out_path, path1, path2)
    logger.info(f'path: {out_path}/{path1}/{path2}')
    logger.info(f'hparam: {hyparam_string}')
    logger.info(li_args)

    tbl = TensorBoardLogger(save_dir=out_path, name=path1, version=path2)
    checkpoint_callback = cb.ModelCheckpoint(monitor="val/mean_ce_loss", mode="min", save_last=True)
    early_stopping = cb.EarlyStopping(monitor="val/mean_ce_loss", mode="min", patience=20)
    trainer = pl.Trainer(max_epochs=args["epochs"], gpus=1, logger=tbl, callbacks=[checkpoint_callback, early_stopping])
    trainer.fit(model, data_module)

if __name__ == '__main__':
    main()
