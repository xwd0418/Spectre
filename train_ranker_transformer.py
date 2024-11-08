import logging, os, sys, torch
import random, pickle
import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.callbacks as cb

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import CSVLogger

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch.distributed as dist

from models.ranked_transformer import HsqcRankedTransformer
from models.ranked_resnet import HsqcRankedResNet
from models.optional_input_ranked_transformer import OptionalInputRankedTransformer
# from models.ranked_double_transformer import DoubleTransformer
from datasets.hsqc_folder_dataset import FolderDataModule
from datasets.oneD_dataset import OneDDataModule
from datasets.optional_2d_folder_dataset import OptionalInputDataModule
from utils.constants import ALWAYS_EXCLUDE, GROUPS, EXCLUDE_FROM_MODEL_ARGS, get_curr_time

import argparse
from argparse import ArgumentParser
from functools import reduce
from datasets.dataset_utils import specific_radius_mfp_loader


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

def data_mux(parser, model_type, data_src, FP_choice, batch_size, ds, args):
    """
        constructs data module based on model_type, and also outputs dimensions of dummy data
        (for graph visualization)
    """
    choice = data_src
    kwargs = vars(parser.parse_args())

    if args['optional_inputs']:
        return OptionalInputDataModule(dir=choice, FP_choice=FP_choice, input_src=["HSQC", "oneD_NMR"], batch_size=batch_size, parser_args=kwargs)
    # OneD datamodule is somewhat buggy, so we will not use hsqc_folder_dataset.py 
    # if args['only_oneD_NMR']:
    #     # oned_dir = "/workspace/OneD_Only_Dataset"
    #     return OneDDataModule(dir=choice, FP_choice=FP_choice, batch_size=batch_size, parser_args=kwargs) 
    # if model_type == "double_transformer": # wangdong: not using it
    #     return FolderDataModule(dir=choice, FP_choice=FP_choice, input_src=["HSQC", "MS"], batch_size=batch_size, parser_args=kwargs )
    elif model_type == "hsqc_transformer":
        return FolderDataModule(dir=choice, FP_choice=FP_choice, input_src=["HSQC"], batch_size=batch_size, parser_args=kwargs)
    elif model_type == "CNN":
        num_channels = kwargs['num_input_channels']
        return FolderDataModule(dir=choice, FP_choice=FP_choice, input_src=[f"HSQC_images_{num_channels}channel"], batch_size=batch_size, parser_args=kwargs)
    elif model_type == "ms_transformer":
        return FolderDataModule(dir=choice, FP_choice=FP_choice, input_src=["MS"], batch_size=batch_size, parser_args=kwargs)
    elif model_type == "transformer_2d1d":
        if kwargs['use_oneD_NMR_no_solvent']:
            return FolderDataModule(dir=choice, FP_choice=FP_choice, input_src=["HSQC", "oneD_NMR"], batch_size=batch_size, parser_args=kwargs)
        else:
            return FolderDataModule(dir=choice, FP_choice=FP_choice, input_src=["HSQC"], batch_size=batch_size, parser_args=kwargs)
    
    raise(f"No datamodule for model type {model_type}.")

def apply_args(parser, model_type):
    if model_type == "hsqc_transformer" or model_type == "ms_transformer" or model_type == "transformer_2d1d":
        HsqcRankedTransformer.add_model_specific_args(parser)
    elif model_type == "CNN":
        HsqcRankedResNet.add_model_specific_args(parser)
   
    else:
        raise(f"No model for model type {model_type}.")

def model_mux(parser, model_type, weights_path, freeze, args):
    logger = logging.getLogger('logging')
    kwargs = vars(parser.parse_args())
    ranking_set_type = kwargs["FP_choice"] 
    kwargs["ranking_set_path"] = f"/workspace/ranking_sets_cleaned_by_inchi/SMILES_{ranking_set_type}_ranking_sets_only_all_info_molecules/val/rankingset.pt"   
   
    for v in EXCLUDE_FROM_MODEL_ARGS:
        if v in kwargs:
            del kwargs[v]

    model_class = None
    if model_type == "hsqc_transformer" or model_type == "ms_transformer" or model_type == "transformer_2d1d":
        if args['optional_inputs']:
            model_class = OptionalInputRankedTransformer
        else:
            model_class = HsqcRankedTransformer
    
    elif model_type == "CNN":
        model_class = HsqcRankedResNet
        
    else:
        raise(f"No model for model type {model_type}.")
    
    
    # elif model_type == "double_transformer":
    #     model_class = DoubleTransformer

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
    
def main(optuna_params=None):
    torch.set_float32_matmul_precision('medium')

    def str2bool(v):    
        # specifically used for arg-paser with boolean values
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')    

    # dependencies: hyun_fp_data, hyun_pair_ranking_set_07_22
    parser = ArgumentParser(add_help=True)
    parser.add_argument("modelname", type=str)
    parser.add_argument("--name_type", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--foldername", type=str, default=f"lightning_logs")
    parser.add_argument("--expname", type=str, default=f"experiment")
    parser.add_argument("--datasrc", type=str, default=f"/workspace/SMILES_dataset")
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--ds", type=str, default="")
    parser.add_argument("--num_workers", type=int, default=16)
    # for early stopping/model saving
    parser.add_argument("--metric", type=str, default="val/mean_rank_1")
    parser.add_argument("--metricmode", type=str, default="max")

    parser.add_argument("--load_all_weights", type=str, default="")
    parser.add_argument("--freeze", type=lambda x:bool(str2bool(x)), default=False)
    parser.add_argument("--validate", type=lambda x:bool(str2bool(x)), default=False)
    parser.add_argument("--test", type=lambda x:bool(str2bool(x)), default=False)
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to the checkpoint file to resume training")

    # different versions of input/output
    # parser.add_argument("--do_hyun_FP", action='store_true', help="use HYUN_FP, otherwise use default R2-6144FP")
    parser.add_argument("--FP_choice", type=str, default="R2-6144FP", help="use which fingerprint as ground truth, default: r2-6144fp") 
    parser.add_argument("--normalize_hsqc", action='store_true', help="input hsqc coordinates will be normalized")
    parser.add_argument("--disable_solvent", action='store_true', help="zero-pad solvent tensor")
    # parser.add_argument("--disable_hsqc_peaks", action='store_true', help="zero-pad hsqc peaks tensor")
    # parser.add_argument("--disable_hsqc_intensity", action='store_true', help="hsqc peaks tensor will be +/-1")    
    parser.add_argument("--enable_hsqc_delimeter_only_2d", action='store_true', 
                        help="add start and end token for hsqc. this flag will be used with only 2d hsqc tensor input")
    parser.add_argument("--use_peak_values",  type=lambda x:bool(str2bool(x)), default=False, help="use peak values in addition to peak signs")
    parser.add_argument("--use_oneD_NMR_no_solvent",  type=lambda x:bool(str2bool(x)), default=True, help="use 1D NMR data")
    parser.add_argument("--rank_by_soft_output",  type=lambda x:bool(str2bool(x)), default=True, help="rank by soft output instead of binary output")
    parser.add_argument("--use_MW",  type=lambda x:bool(str2bool(x)), default=True, help="using mass spectra")
    parser.add_argument("--use_Jaccard",  type=lambda x:bool(str2bool(x)), default=False, help="using Jaccard similarity instead of cosine similarity")
    parser.add_argument("--jittering",  type=str, default="None", help="a data augmentation technique that jitters the peaks. Choose 'normal' or 'uniform' to choose jittering distribution" )
    
    # count-based FP
    parser.add_argument("--num_class",  type=int, default=25, help="size of CE label class when using count based FP")
    parser.add_argument("--loss_func",  type=str, default="MSE", help="either MSE or CE")
    
    # optional 2D input
    parser.add_argument("--optional_inputs",  type=lambda x:bool(str2bool(x)), default=False, help="use optional 2D input, inference will contain different input versions")
    parser.add_argument("--combine_oneD_only_dataset",  type=lambda x:bool(str2bool(x)), default=False, help="use molecules with only 1D input")
    parser.add_argument("--only_oneD_NMR",  type=lambda x:bool(str2bool(x)), default=False, help="only use oneD NMR, C or H or both. By default is both")
    parser.add_argument("--only_C_NMR",  type=lambda x:bool(str2bool(x)), default=False, help="only use oneD C_NMR. Need to use together with only_oneD_NMR")
    parser.add_argument("--only_H_NMR",  type=lambda x:bool(str2bool(x)), default=False, help="only use oneD H_NMR. Need to use together with only_oneD_NMR")
    parser.add_argument("--separate_classifier",  type=lambda x:bool(str2bool(x)), default=False, help="use separate classifier for various 2D/1D input")
    parser.add_argument("--weighted_sample_based_on_input_type",  type=lambda x:bool(str2bool(x)), default=False, help="use weighted loss based on input type")
    parser.add_argument("--sampling_strategy",  type=str, default="none", help="sampling strategy for weighted loss")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--train_on_all_info_set", type=lambda x:bool(str2bool(x)), default=False)
    
    # entropy based FP
    parser.add_argument("--FP_building_type", type=str, default="Normal", help="Normal or Exact")
    parser.add_argument("--out_dim", type=int, default="6144", help="the size of output fingerprint to be predicted")
    
    args = vars(parser.parse_known_args()[0])
    # if args['only_C_NMR'] or args['only_H_NMR']:
    #     assert args['only_oneD_NMR'], "only_C_NMR or only_H_NMR should be used with only_oneD_NMR"
    # if args['only_oneD_NMR']:
    #     assert args['combine_oneD_only_dataset'], "oneD_NMR live in both datasets"
    if args['weighted_sample_based_on_input_type']:
        assert args['combine_oneD_only_dataset'] and args['optional_inputs'], "Only available for combined dataset"
    
    if args['FP_choice'].startswith("pick_entropy"): # should be in the format of "pick_entropy_r9"
            only_2d = not args['use_oneD_NMR_no_solvent']
            FP_building_type = args['FP_building_type'].split("_")[-1]
            specific_radius_mfp_loader.setup(only_2d=only_2d,FP_building_type=FP_building_type, out_dim=args['out_dim'])
            specific_radius_mfp_loader.set_max_radius(int(args['FP_choice'].split("_")[-1][1:]), only_2d=only_2d)
    
    seed_everything(seed=args["random_seed"])   
    
    # general args
    apply_args(parser, args["modelname"])

    # Model args
    args_with_model = vars(parser.parse_known_args()[0])
    li_args = list(args_with_model.items())
    if args['foldername'] == "debug":
        args["epochs"] = 2

    # Tensorboard setup
    # curr_exp_folder_name = 'NewRepoNewDataOldCode'
    curr_exp_folder_name = "datasetV4_no_oneD_snooping"
    out_path       =      f"/workspace/reproduce_previous_works/{curr_exp_folder_name}"
    # out_path =            f"/root/MorganFP_prediction/reproduce_previous_works/{curr_exp_folder_name}"
    out_path_final =      f"/root/MorganFP_prediction/reproduce_previous_works/{curr_exp_folder_name}"
    os.makedirs(out_path_final, exist_ok=True)
    os.makedirs(out_path, exist_ok=True)
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
    # my_logger.info(f'[Main] using GPU : {torch.cuda.get_device_name()}')
    
    # Model and Data setup
    model = model_mux(parser, args["modelname"], args["load_all_weights"], args["freeze"], args)
    from pytorch_lightning.utilities.model_summary import summarize
    my_logger.info(f"[Main] Model Summary: {summarize(model)}")
    
    data_module = data_mux(parser, args["modelname"], args["datasrc"], args["FP_choice"], args["bs"], args["ds"], args)
    tbl = TensorBoardLogger(save_dir=out_path, name=path1, version=path2)

    # Trainer, callbacks
    metric, metricmode, patience = args["metric"], args["metricmode"], args["patience"]
    if args['optional_inputs']:
        checkpoint_callbacks = []
        my_logger.info("[Main] Using Optional Input")
        # metric = "val_mean_rank_1/all_nmr_combination_avg" # essientially the same as mean_rank_1, just naming purposes
        for metric in  ["all_inputs", "HSQC_H_NMR", "HSQC_C_NMR", "only_hsqc", "only_1d", "only_H_NMR",  "only_C_NMR"]:
            checkpoint_callbacks.append(cb.ModelCheckpoint(monitor=f"val_mean_rank_1/{metric}", mode=metricmode,
                                                           filename= "{epoch}-"+metric)) 
    else:
        checkpoint_callbacks =[cb.ModelCheckpoint(monitor=metric, mode=metricmode, save_last=True, save_top_k = 1)]
        
    early_stop_metric = 'val_mean_rank_1/all_inputs' if args['optional_inputs'] else args["metric"]
    early_stopping = EarlyStopping(monitor=early_stop_metric, mode=metricmode, patience=patience)
    lr_monitor = cb.LearningRateMonitor(logging_interval="step")
    trainer = pl.Trainer(
                         max_epochs=args["epochs"],
                         accelerator="auto",
                         logger=tbl, 
                         callbacks=[early_stopping, lr_monitor]+checkpoint_callbacks,
                        )
    if args["validate"]:
        my_logger.info("[Main] Just performing validation step")
        trainer.validate(model, data_module)
    elif args['test']:
        my_logger.info("[Main] Just performing test step")
        raise Exception("should use test_on_all_info_subset.ipynb")
        
    else:
        try:
            my_logger.info("[Main] Begin Training!")
            trainer.fit(model, data_module,ckpt_path=args["checkpoint_path"])

            model.change_ranker_for_testing()
            checkpoint_callback = checkpoint_callbacks[0]
            if not args['optional_inputs']:
                my_logger.info(f"[Main] Testing path {checkpoint_callback.best_model_path}!")
                test_result = trainer.test(model, data_module,ckpt_path=checkpoint_callback.best_model_path)
                test_result[0]['best_epoch'] = checkpoint_callback.best_model_path.split("/")[-1].split("-")[0]
                # save test result as pickle
                with open(f"{out_path}/{path1}/{path2}/test_result.pkl", "wb") as f:
                    pickle.dump(test_result, f)
            else:
                # loader_all_inputs, loader_HSQC_H_NMR, loader_HSQC_C_NMR, loader_only_hsqc, loader_only_1d, loader_only_H_NMR, loader_only_C_NMR
                data_module.setup("test")
                all_7_dataloaders = data_module.test_dataloader()
                all_test_results = [{}]
                for loader_idx, (checkpoint_callback, curr_dataloader) in enumerate(zip(checkpoint_callbacks, all_7_dataloaders)):                               
                    model.only_test_this_loader(loader_idx=loader_idx)
                    my_logger.info(f"[Main] Testing path {checkpoint_callback.best_model_path}!")
                    test_result = trainer.test(model, data_module,ckpt_path=checkpoint_callback.best_model_path)
                    test_result[0][f'best_epoch_{checkpoint_callback.monitor.split("/")[-1]}'] = checkpoint_callback.best_model_path.split("/")[-1].split("-")[0]
                    all_test_results[0].update(test_result[0])
                # save test result as pickle
                with open(f"{out_path}/{path1}/{path2}/test_result.pkl", "wb") as f:
                    pickle.dump(all_test_results, f)
            
        except Exception as e:
            my_logger.error(f"[Main] Error: {e}")
            raise(e)
        finally: #Finally move all content from out_path to out_path_final
            my_logger.info("[Main] Done!")
            my_logger.info("[Main] test result: \n")
            # my_logger.info(f"{test_result}")
            for key, value in test_result[0].items():
                my_logger.info(f"{key}: {value}")
            os.system(f"cp -r {out_path}/* {out_path_final}/ && rm -rf {out_path}/*")
            logging.shutdown()

    # return test_result[0]['test/mean_rank_1'] # for optuna with non-flexble model
    # return test_result[0]['test/mean_rank_1_all_inputs'] # for optuna with non-flexble model
        


if __name__ == '__main__':
    
    main()
