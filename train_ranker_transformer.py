import logging, os, sys, torch
import pytorch_lightning as pl
import pytorch_lightning.callbacks as cb

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch.distributed as dist

from models.ranked_transformer import HsqcRankedTransformer
# from models.ranked_double_transformer import DoubleTransformer
from datasets.hsqc_folder_dataset import FolderDataModule
from utils.constants import ALWAYS_EXCLUDE, GROUPS, EXCLUDE_FROM_MODEL_ARGS, get_curr_time

from argparse import ArgumentParser
from functools import reduce

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
    # limited hyperparameter experiment name, all hyperparameter string, expname + time
    return f"{expname}_[{get_curr_time()}]_[{'_'.join(hierarchical)}]", '_'.join(hierarchical_unlimited), f"{expname}_[{get_curr_time()}]"

def data_mux(parser, model_type, data_src, do_hyun_fp, batch_size, ds):
    """
        constructs data module based on model_type, and also outputs dimensions of dummy data
        (for graph visualization)
    """
    choice = data_src

    if model_type == "double_transformer":
        return FolderDataModule(dir=choice, do_hyun_fp=do_hyun_fp, input_src=["HSQC", "MS"], batch_size=batch_size)
    elif model_type == "hsqc_transformer":
        return FolderDataModule(dir=choice, do_hyun_fp=do_hyun_fp, input_src=["HSQC"], batch_size=batch_size)
    elif model_type == "ms_transformer":
        return FolderDataModule(dir=choice, do_hyun_fp=do_hyun_fp, input_src=["MS"], batch_size=batch_size)
    raise(f"No datamodule for model type {model_type}.")

def apply_args(parser, model_type):
    if model_type == "hsqc_transformer" or model_type == "ms_transformer":
        HsqcRankedTransformer.add_model_specific_args(parser)
    # elif model_type == "double_transformer":
    #     DoubleTransformer.add_model_specific_args(parser)
    else:
        raise(f"No model for model type {model_type}.")

def model_mux(parser, model_type, weights_path, freeze):
    logger = logging.getLogger('logging')
    kwargs = vars(parser.parse_args())
    ranking_set_type = "HYUN_FP" if kwargs["do_hyun_FP"] else "R2_6144FP"
    kwargs["ranking_set_path"] = f"/root/MorganFP_prediction/reproduce_previous_works/smart4.5/ranking_sets/SMILES_{ranking_set_type}_ranking_sets/val/rankingset.pt"
        
    for v in EXCLUDE_FROM_MODEL_ARGS:
        if v in kwargs:
            del kwargs[v]

    model_class = None
    if model_type == "hsqc_transformer" or model_type == "ms_transformer":
        model_class = HsqcRankedTransformer
    # elif model_type == "double_transformer":
    #     model_class = DoubleTransformer
    else:
        raise(f"No model for model type {model_type}.")

    if weights_path: # initialize with loaded state if non-empty string passed
        model = model_class.load_from_checkpoint(weights_path, strict=False)
        logger.info("[Main] Loading model from Weights")
    else: # or from scratch
        model = model_class(**kwargs)
        logger.info("[Main] Freshly initializing model")

    if freeze:
        logger.info("[Main] Freezing Model Weight")
        for param in model.parameters():
            param.requires_grad = False
    return model

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
    parser.add_argument("--name_type", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--foldername", type=str, default=f"lightning_logs")
    parser.add_argument("--expname", type=str, default=f"experiment")
    parser.add_argument("--datasrc", type=str, default=f"/root/MorganFP_prediction/James_dataset_zips/SMILES_dataset")
    parser.add_argument("--do_hyun_FP", action='store_true', help="use HYUN_FP, otherwise use default R2-6144FP")
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--ds", type=str, default="")
    # for early stopping/model saving
    parser.add_argument("--metric", type=str, default="val/mean_ce_loss")
    parser.add_argument("--metricmode", type=str, default="min")

    parser.add_argument("--load_all_weights", type=str, default="")
    parser.add_argument("--freeze", type=bool, default=False)
    parser.add_argument("--validate", type=bool, default=False)
    args = vars(parser.parse_known_args()[0])

    # general args
    apply_args(parser, args["modelname"])

    # Model args
    args_with_model = vars(parser.parse_known_args()[0])
    li_args = list(args_with_model.items())

    # Tensorboard setup
    out_path = "/root/MorganFP_prediction/reproduce_previous_works/reproduce_results_transformer"
    exp_name, hparam_string, exp_time_string = exp_string(args["expname"], li_args)
    path1 = args["foldername"]
    if args["name_type"] == 0: # full hyperparameter string
        path2 = exp_name
    elif args["name_type"] == 1: # only experiment name and time
        path2 = exp_time_string
    else: # only experiment name parameter
        path2 = args["expname"]

    # Logger setup
    my_logger = init_logger(out_path, path1, path2)
    my_logger.info(f'[Main] Output Path: {out_path}/{path1}/{path2}')
    my_logger.info(f'[Main] Hyperparameters: {hparam_string}')

    # Model and Data setup
    model = model_mux(parser, args["modelname"], args["load_all_weights"], args["freeze"])
    data_module = data_mux(parser, args["modelname"], args["datasrc"], args["do_hyun_FP"], args["bs"], args["ds"])

    # Trainer, callbacks
    metric, metricmode, patience = args["metric"], args["metricmode"], args["patience"]

    tbl = TensorBoardLogger(save_dir=out_path, name=path1, version=path2)
    checkpoint_callback = cb.ModelCheckpoint(monitor=metric, mode=metricmode, save_last=True, save_top_k = 3)
    early_stopping = EarlyStopping(monitor=metric, mode=metricmode, patience=patience)
    lr_monitor = cb.LearningRateMonitor(logging_interval="step")
    trainer = pl.Trainer(max_epochs=args["epochs"], accelerator="gpu", logger=tbl, callbacks=[checkpoint_callback, early_stopping, lr_monitor])
    if args["validate"]:
        my_logger.info("[Main] Just performing validation step")
        trainer.validate(model, data_module)
    else:
        my_logger.info("[Main] Begin Training!")
        trainer.fit(model, data_module)
        my_logger.info("[Main] Final Validation Step:")
        if dist.is_initialized():
            rank = dist.get_rank()
            if rank == 0:
                validation_trainer = pl.Trainer(accelerator="gpu", devices=1)
                validation_trainer.validate(model, data_module)
        # validation_trainer.test(model, datamodule=??)
    my_logger.info("[Main] Done Training!")

if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    main()
