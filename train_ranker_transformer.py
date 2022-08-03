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

def data_mux(parser, model_type, do_hyun_fp, input_src, batch_size):
    my_dir = "/workspace/smart4.5/tempdata/hyun_fp_data/hsqc_ms_pairs"
    return FolderDataModule(dir=my_dir, do_hyun_fp=do_hyun_fp, input_src=input_src, batch_size=batch_size)

def register_parsers(parser):
    sub = parser.add_subparsers(help="Sub-commands help")

    dbl_tnsfm = sub.add_parser("double_transformer", help="Train a model")
    DoubleTransformer.add_model_specific_args(dbl_tnsfm)
    HsqcRankedTransformer.add_model_specific_args(dbl_tnsfm, "hsqc")
    HsqcRankedTransformer.add_model_specific_args(dbl_tnsfm, "ms")

    return sub

def model_mux(parser, model_type):
    if model_type == "hsqc_transformer" or model_type == "ms_transformer":
        HsqcRankedTransformer.add_model_specific_args(parser)
        args = vars(parser.parse_args())
        return HsqcRankedTransformer(*args) 
    elif model_type == "double_transformer":
        DoubleTransformer.add_model_specific_args(parser)
        HsqcRankedTransformer.add_model_specific_args(parser, "hsqc")
        HsqcRankedTransformer.add_model_specific_args(parser, "ms")
        hsqc = HsqcRankedTransformer.prune_args(vars(parser.parse_args()), "hsqc")
        hsqc_transformer = HsqcRankedTransformer(**hsqc)
        ms = HsqcRankedTransformer.prune_args(vars(parser.parse_args()), "ms")
        ms_transformer = HsqcRankedTransformer(**ms)
        print("hsqc_args", hsqc)
        print("ms_args", ms)
        kwargs = vars(parser.parse_args())
        return DoubleTransformer(hsqc_transformer=hsqc_transformer, spec_transformer=ms_transformer, **kwargs) 
    else:
        raise(f"Model {model_type} does not exist")

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
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--expname", type=str, default=f"experiment")
    parser.add_argument("--foldername", type=str, default=f"lightning_logs")
    args, _ = parser.parse_known_args()
    args = vars(args)

    model = model_mux(parser, args["modelname"])
    data_module = data_mux(parser, args["modelname"], True, ["HSQC", "MS"], 64)
    print(parser.parse_args())

    li_args = list(args.items())
    hyparam_string = "_".join([f"{hyparam}={val}"for hyparam, val in li_args if hyparam not in ["modelname", "expname", "foldername"]])
    out_path = "/data/smart4.5"
    path1, path2 = args["foldername"], f"{args['expname']}_{get_curr_time()}_{hyparam_string}"

    logger = init_logger(out_path, path1, path2)
    logger.info(hyparam_string)
    logger.info(li_args)

    tbl = TensorBoardLogger(save_dir=out_path, name=path1, version=path2)
    checkpoint_callback = cb.ModelCheckpoint(monitor="val/mean_ce_loss", mode="min", save_last=True)
    trainer = pl.Trainer(max_epochs=args["epochs"], gpus=1, logger=tbl, callbacks=[checkpoint_callback])
    trainer.fit(model, data_module)

if __name__ == '__main__':
    main()
