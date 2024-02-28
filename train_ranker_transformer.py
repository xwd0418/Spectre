import logging, os, sys, torch
import random
import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.callbacks as cb

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import CSVLogger

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

def data_mux(parser, model_type, data_src, FP_choice, batch_size, ds):
    """
        constructs data module based on model_type, and also outputs dimensions of dummy data
        (for graph visualization)
    """
    choice = data_src
    kwargs = vars(parser.parse_args())

    if model_type == "double_transformer":
        return FolderDataModule(dir=choice, FP_choice=FP_choice, input_src=["HSQC", "MS"], batch_size=batch_size, parser_args=kwargs )
    elif model_type == "hsqc_transformer":
        return FolderDataModule(dir=choice, FP_choice=FP_choice, input_src=["HSQC"], batch_size=batch_size, parser_args=kwargs)
    elif model_type == "ms_transformer":
        return FolderDataModule(dir=choice, FP_choice=FP_choice, input_src=["MS"], batch_size=batch_size, parser_args=kwargs)
    elif model_type == "transformer_2d1d":
        if kwargs['use_oneD_NMR_no_solvent']:
            return FolderDataModule(dir=choice, FP_choice=FP_choice, input_src=["HSQC", "oneD_NMR"], batch_size=batch_size, parser_args=kwargs)
        else:
            return FolderDataModule(dir=choice, FP_choice=FP_choice, input_src=["HSQC", "detailed_oneD_NMR"], batch_size=batch_size, parser_args=kwargs)
    
    raise(f"No datamodule for model type {model_type}.")

def apply_args(parser, model_type):
    if model_type == "hsqc_transformer" or model_type == "ms_transformer" or model_type == "transformer_2d1d":
        HsqcRankedTransformer.add_model_specific_args(parser)
    # elif model_type == "double_transformer":
    #     DoubleTransformer.add_model_specific_args(parser)
    else:
        raise(f"No model for model type {model_type}.")

def model_mux(parser, model_type, weights_path, freeze):
    logger = logging.getLogger('logging')
    kwargs = vars(parser.parse_args())
    ranking_set_type = kwargs["FP_choice"] 
    kwargs["ranking_set_path"] = f"/workspace/ranking_sets_cleaned_by_inchi/SMILES_{ranking_set_type}_ranking_sets/val/rankingset.pt"
   
    for v in EXCLUDE_FROM_MODEL_ARGS:
        if v in kwargs:
            del kwargs[v]

    model_class = None
    if model_type == "hsqc_transformer" or model_type == "ms_transformer" or model_type == "transformer_2d1d":
        model_class = HsqcRankedTransformer
    # elif model_type == "double_transformer":
    #     model_class = DoubleTransformer
    else:
        raise(f"No model for model type {model_type}.")

    if weights_path: # initialize with loaded state if non-empty string passed
        model = model_class.load_from_checkpoint(weights_path, strict=False)
        logger.info("[Main] Loading model from Weights")
    else: # or from scratch
        print(kwargs['modelname'])
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

def seed_everything(seed):
    """
    Set the random seed for reproducibility.
    """
    pl.seed_everything(seed,  workers=True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed)
    random.seed(seed)
    # torch.use_deterministic_algorithms(True)
    
def main():
    seed_everything(seed=2024)
    
    # dependencies: hyun_fp_data, hyun_pair_ranking_set_07_22
    parser = ArgumentParser(add_help=True)
    parser.add_argument("modelname", type=str)
    parser.add_argument("--name_type", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--foldername", type=str, default=f"lightning_logs")
    parser.add_argument("--expname", type=str, default=f"experiment")
    parser.add_argument("--datasrc", type=str, default=f"/workspace/SMILES_dataset")
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--ds", type=str, default="")
    parser.add_argument("--num_workers", type=int, default=16)
    # for early stopping/model saving
    parser.add_argument("--metric", type=str, default="val/mean_rank_1")
    parser.add_argument("--metricmode", type=str, default="max")

    parser.add_argument("--load_all_weights", type=str, default="")
    parser.add_argument("--freeze", type=bool, default=False)
    parser.add_argument("--validate", type=bool, default=False)
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to the checkpoint file to resume training")

    # different versions of input/output
    # parser.add_argument("--do_hyun_FP", action='store_true', help="use HYUN_FP, otherwise use default R2-6144FP")
    parser.add_argument("--FP_choice", type=str, default="R2-6144FP", help="use which fingerprint as ground truth, default: r2-6144fp") 
    parser.add_argument("--normalize_hsqc", action='store_true', help="input hsqc coordinates will be normalized")
    parser.add_argument("--disable_solvent", action='store_true', help="zero-pad solvent tensor")
    parser.add_argument("--disable_hsqc_peaks", action='store_true', help="zero-pad hsqc peaks tensor")
    parser.add_argument("--disable_hsqc_intensity", action='store_true', help="hsqc peaks tensor will be +/-1")    
    parser.add_argument("--enable_hsqc_delimeter_only_2d", action='store_true', 
                        help="add start and end token for hsqc. this flag will be used with only 2d hsqc tensor input")
    
    parser.add_argument("--use_oneD_NMR_no_solvent",  type=bool, default=True, help="use detailed 1D NMR data")
    parser.add_argument("--rank_by_soft_output",  type=bool, default=True, help="rank by soft output instead of binary output")
    parser.add_argument("--use_MW",  type=bool, default=True, help="using mass spectra")
    
    args = vars(parser.parse_known_args()[0])

    # general args
    apply_args(parser, args["modelname"])

    # Model args
    args_with_model = vars(parser.parse_known_args()[0])
    li_args = list(args_with_model.items())

    # Tensorboard setup
    out_path = "/root/MorganFP_prediction/reproduce_previous_works/compute_ranking_lower_bound"
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
    # # try:
    # import torch._dynamo
    # torch._dynamo.reset()
    # compiled_model = torch.compile(model, mode="reduce-overhead")
    # model = compiled_model
    # my_logger.info("[Main] Compiled Model")
    # # except:
    #     # my_logger.info("[Main] Compile failed, continuing without compilation")
    data_module = data_mux(parser, args["modelname"], args["datasrc"], args["FP_choice"], args["bs"], args["ds"])

    # Trainer, callbacks
    metric, metricmode, patience = args["metric"], args["metricmode"], args["patience"]

    tbl = TensorBoardLogger(save_dir=out_path, name=path1, version=path2)
    checkpoint_callback = cb.ModelCheckpoint(monitor=metric, mode=metricmode, save_last=True, save_top_k = 1)
  
    early_stopping = EarlyStopping(monitor=metric, mode=metricmode, patience=patience)
    lr_monitor = cb.LearningRateMonitor(logging_interval="step")
    trainer = pl.Trainer(
                         max_epochs=args["epochs"],
                         accelerator="gpu",
                         logger=tbl, 
                         callbacks=[checkpoint_callback, early_stopping, lr_monitor],
                        )
    if args["validate"]:
        my_logger.info("[Main] Just performing validation step")
        trainer.validate(model, data_module)
    else:
        my_logger.info("[Main] Begin Training!")
        trainer.fit(model, data_module,ckpt_path=args["checkpoint_path"])
        # if dist.is_initialized():
        #     my_logger.info("[Main] Begin Testing:")
        #     rank = dist.get_rank()
        #     if rank == 0: # To only run the test once
        #         model.change_ranker_for_testing()
        #         # testlogger = CSVLogger(save_dir=out_path, name=path1, version=path2)

        #         test_trainer = pl.Trainer(accelerator="gpu", logger=tbl, devices=1,)
        #         test_trainer.test(model, data_module,ckpt_path=checkpoint_callback.best_model_path )
        #         # test_trainer.test(model, data_module,ckpt_path=checkpoint_callback.last_model_path )
        model.change_ranker_for_testing()
        my_logger.info(f"[Main] Testing path {checkpoint_callback.best_model_path}!")
        trainer.test(model, data_module,ckpt_path=checkpoint_callback.best_model_path)
    my_logger.info("[Main] Done!")
    try:
        my_logger.info(f'[Main] using GPU : {torch.cuda.get_device_name()}')
    except:
        my_logger.info(f'[Main] could not find GPU name')

if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    main()
