import torch, pytorch_lightning as pl
import pytorch_lightning.callbacks as cb
from argparse import ArgumentParser
from functools import reduce
from pytorch_lightning.loggers import TensorBoardLogger

from models.ranked_transformer import HsqcRankedTransformer
from models.ranked_double_transformer import DoubleTransformer
from datasets.hsqc_folder_dataset import FolderDataModule
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import logging, os, sys
from datetime import datetime
from pytz import timezone

def get_curr_time():
    pst = timezone("PST8PDT")
    california_time = datetime.now(pst)
    return california_time.strftime("%m_%d_%Y_%H:%M")

ALWAYS_EXCLUDE = ["modelname", "debug", "expname", "foldername", "datasrc", "patience", "ds"] + ["hsqc_weights", "ms_weights"]
GROUPS = [
    set(["lr", "epochs", "bs"]),
    set(["heads", "layers", "hsqc_heads", "hsqc_layers", "ms_heads", "ms_layers"])
]
def exp_string(expname, args):
    """
        Gets an experiment string with a format (expname_[time started]_[some hyperparameters])
    """
    def stringify(items, limit=True):
        # max 8 params
        if limit:
            return "_".join(map(lambda x : f'{x[0]}={x[1]}', sorted(list(items), key=lambda x : x[0])[:8]))
        else:
            return "_".join(map(lambda x : f'{x[0]}={x[1]}', sorted(list(items), key=lambda x : x[0])))
    all_grouped = set(reduce(lambda x, y: x.union(y), GROUPS))
    filtered = [(hyparam, val) for hyparam, val in args if hyparam not in ALWAYS_EXCLUDE]
    
    grouped_params = [stringify(filter(lambda x: x[0] in g, filtered)) for g in GROUPS]
    ungrouped_params = [stringify(filter(lambda x: x[0] not in all_grouped, filtered))]
    ungrouped_params_unlimited = [stringify(filter(lambda x: x[0] not in all_grouped, filtered), limit=False)]

    hierarchical = grouped_params + ungrouped_params
    hierarchical_unlimited = grouped_params + ungrouped_params_unlimited
    return f"{expname}_[{get_curr_time()}]_[{'_'.join(hierarchical)}]", '_'.join(hierarchical_unlimited)

def data_mux(parser, model_type, data_src, do_hyun_fp, batch_size, ds):
    """
        constructs data module based on model_tyle, and also outputs dimensions of dummy data
        (for graph visualization)
    """
    my_dir = f"/workspace/smart4.5/tempdata/hyun_fp_data/{data_src}"
    signed_dir = f"/workspace/smart4.5/tempdata/bounded_hyun_fp_data/{data_src}"
    if model_type == "double_transformer":
        return FolderDataModule(dir=my_dir, do_hyun_fp=do_hyun_fp, input_src=["HSQC", "MS"], batch_size=batch_size)
    elif model_type == "hsqc_transformer":
        if ds == "signed_intensity":
            print("==== USING SIGNED HSQC DATA ====")
            return FolderDataModule(dir=signed_dir, do_hyun_fp=do_hyun_fp, input_src=["HSQC"], batch_size=batch_size)
        return FolderDataModule(dir=my_dir, do_hyun_fp=do_hyun_fp, input_src=["HSQC"], batch_size=batch_size)
    elif model_type == "ms_transformer":
        return FolderDataModule(dir=my_dir, do_hyun_fp=do_hyun_fp, input_src=["MS"], batch_size=batch_size)
    raise(f"No datamodule for model type {model_type}.")

def apply_args(parser, model_type):
    if model_type == "hsqc_transformer" or model_type == "ms_transformer":
        HsqcRankedTransformer.add_model_specific_args(parser)
    elif model_type == "double_transformer":
        DoubleTransformer.add_model_specific_args(parser)
    else:
        raise(f"No model for model type {model_type}.")
def model_mux(parser, model_type):
    eliminate = ["modelname", "debug", "expname", "foldername", "datasrc"]
    if model_type == "hsqc_transformer" or model_type == "ms_transformer":
        kwargs = vars(parser.parse_args())
        for v in eliminate:
            del kwargs[v]
        return HsqcRankedTransformer(**kwargs) 
    elif model_type == "double_transformer":
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
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--ds", type=str, default="")
    args = vars(parser.parse_known_args()[0])

    apply_args(parser, args["modelname"])
    data_module = data_mux(parser, args["modelname"], args["datasrc"], True, args["bs"], args["ds"])
    args_with_model = vars(parser.parse_known_args()[0])
    li_args = list(args_with_model.items())

    out_path, (exp_name, hparam_string)  = "/data/smart4.5", exp_string(args["expname"], li_args)
    path1, path2 = args["foldername"], args["expname"] if args["debug"] else exp_name

    patience = args["patience"]

    logger = init_logger(out_path, path1, path2)
    logger.info(f'path: {out_path}/{path1}/{path2}')
    logger.info(f'hparam: {hparam_string}')

    model = model_mux(parser, args["modelname"])

    tbl = TensorBoardLogger(save_dir=out_path, name=path1, version=path2)
    checkpoint_callback = cb.ModelCheckpoint(monitor="val/mean_ce_loss", mode="min", save_last=True)
    early_stopping = EarlyStopping(monitor="val/mean_ce_loss", mode="min", patience=patience)
    lr_monitor = cb.LearningRateMonitor(logging_interval="step")
    trainer = pl.Trainer(max_epochs=args["epochs"], gpus=1, logger=tbl, callbacks=[checkpoint_callback, early_stopping, lr_monitor])
    trainer.fit(model, data_module)

if __name__ == '__main__':
    main()
