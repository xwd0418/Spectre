import pathlib
import yaml

DATASET_root_path = pathlib.Path("/workspace/")
curr_exp_folder_name = "entropy_on_hashes"

import logging, os, sys, torch
import random, pickle
import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.callbacks as cb

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.model_summary import summarize
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
from utils.NP_classwise_accu_plot import plot_result_dict_by_sorted_names

import argparse
from argparse import ArgumentParser
from functools import reduce
from datasets.dataset_utils import  FP_Loader_Configer
fp_loader_configer = FP_Loader_Configer()


import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message="You are using `torch.load` with `weights_only=False`")
warnings.filterwarnings("ignore", category=UserWarning, message="The PyTorch API of nested tensors is in prototype stage and will change in the near future.")



def data_mux(parser, model_type, data_src, FP_choice, batch_size, ds, args, fp_loader):
    """
        constructs data module based on model_type, and also outputs dimensions of dummy data
        (for graph visualization)
    """
    choice = data_src
    kwargs = args # vars(parser.parse_args())

    if args['optional_inputs']:
        return OptionalInputDataModule(dir=choice, FP_choice=FP_choice, input_src=["HSQC", "oneD_NMR"], fp_loader=fp_loader, batch_size=batch_size, parser_args=kwargs)
    # OneD datamodule is somewhat buggy, so we will not use hsqc_folder_dataset.py 
    if args['only_oneD_NMR']:
        # oned_dir = "/workspace/OneD_Only_Dataset"
        # here the choice is still SMILES_dataset, but infact, OneDDataset use both datasets
        return OneDDataModule(dir=choice, FP_choice=FP_choice, fp_loader=fp_loader, batch_size=batch_size, parser_args=kwargs) 
    if model_type == "double_transformer": # wangdong: not using it
        return FolderDataModule(dir=choice, FP_choice=FP_choice, input_src=["HSQC", "MS"], fp_loader=fp_loader, batch_size=batch_size, parser_args=kwargs )
    elif model_type == "hsqc_transformer":
        return FolderDataModule(dir=choice, FP_choice=FP_choice, input_src=["HSQC"], fp_loader=fp_loader, batch_size=batch_size, parser_args=kwargs)
    elif model_type == "CNN":
        num_channels = kwargs['num_input_channels']
        return FolderDataModule(dir=choice, FP_choice=FP_choice, input_src=[f"HSQC_images_{num_channels}channel"], fp_loader=fp_loader, batch_size=batch_size, parser_args=kwargs)
    elif model_type == "ms_transformer":
        return FolderDataModule(dir=choice, FP_choice=FP_choice, input_src=["MS"], fp_loader=fp_loader, batch_size=batch_size, parser_args=kwargs)
    elif model_type == "transformer_2d1d":
        if kwargs['use_oneD_NMR_no_solvent']:
            return FolderDataModule(dir=choice, FP_choice=FP_choice, input_src=["HSQC", "oneD_NMR"], fp_loader=fp_loader, batch_size=batch_size, parser_args=kwargs)
        else:
            return FolderDataModule(dir=choice, FP_choice=FP_choice, input_src=["HSQC"], fp_loader=fp_loader, batch_size=batch_size, parser_args=kwargs)
    
    raise(f"No datamodule for model type {model_type}.")

def apply_args(parser, model_type):
    if model_type == "hsqc_transformer" or model_type == "ms_transformer" or model_type == "transformer_2d1d":
        HsqcRankedTransformer.add_model_specific_args(parser)
    elif model_type == "CNN":
        HsqcRankedResNet.add_model_specific_args(parser)
   
    else:
        raise(f"No model for model type {model_type}.")

def model_mux(parser, model_type, weights_path, freeze, args, fp_loader):
    logger = logging.getLogger('logging')
    kwargs = args.copy() #vars(parser.parse_args())
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
        model = model_class(fp_loader, **kwargs)
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

def add_parser_arguments( parser):
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
        
    parser.add_argument("modelname", type=str)
    parser.add_argument("--name_type", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--foldername", type=str, default=f"lightning_logs")
    parser.add_argument("--expname", type=str, default=f"experiment")
    parser.add_argument("--datasrc", type=str, default=f"/workspace/SMILES_dataset")
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--accumulate_grad_batches_num", type=int, default=4)
        
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--ds", type=str, default="")
    parser.add_argument("--num_workers", type=int, default=4)
    # for early stopping/model saving
    parser.add_argument("--metric", type=str, default="val/mean_cos")
    parser.add_argument("--metricmode", type=str, default="max")

    parser.add_argument("--load_all_weights", type=str, default="")
    parser.add_argument("--freeze", type=lambda x:bool(str2bool(x)), default=False)
    parser.add_argument("--validate", type=lambda x:bool(str2bool(x)), default=False)
    parser.add_argument("--test", type=lambda x:bool(str2bool(x)), default=False)
    parser.add_argument("--test_on_deepsat_retrieval_set", type=lambda x:bool(str2bool(x)), default=False)
    
    parser.add_argument("--debug", type=lambda x:bool(str2bool(x)), default=False)
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to the checkpoint file to resume training")
    parser.add_argument("--delete_checkpoint", type=lambda x:bool(str2bool(x)), default=False, help="Delete the checkpoint file after training")

    # different versions of input/output
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
    parser.add_argument("--jittering",  type=float, default=0, help="a data augmentation technique that jitters the peaks. Choose 'normal' or 'uniform' to choose jittering distribution" )
    parser.add_argument("--rank_by_test_set",  type=lambda x:bool(str2bool(x)), default=False, help="rank by test set instead of entire set. only used during grid search")
    # count-based FP
    parser.add_argument("--num_class",  type=int, default=25, help="size of CE label class when using count based FP")
    parser.add_argument("--loss_func",  type=str, default="MSE", help="either MSE or CE")
    
    # optional 2D input
    parser.add_argument("--optional_inputs",  type=lambda x:bool(str2bool(x)), default=False, help="use optional 2D input, inference will contain different input versions")
    parser.add_argument("--optional_MW",  type=lambda x:bool(str2bool(x)), default=False, help="also make molecular weight as optional input")
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
    parser.add_argument("--out_dim", type=lambda x: int(x) if x.isdigit() else x, default="6144", help="the size of output fingerprint to be predicted. If set to inf, then use all fragements/bits")

    # return test_result[0]['test/mean_rank_1'] # for optuna with non-flexble model
    # return test_result[0]['test/mean_rank_1_all_inputs'] # for optuna with non-flexble model
        
    
if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')

    # dependencies: hyun_fp_data, hyun_pair_ranking_set_07_22
    parser = ArgumentParser(add_help=True)
    add_parser_arguments(parser)
    
    args = vars(parser.parse_known_args()[0])
    apply_args(parser, args["modelname"])
    args = vars(parser.parse_known_args()[0])
    # if args['only_C_NMR'] or args['only_H_NMR']:
    #     assert args['only_oneD_NMR'], "only_C_NMR or only_H_NMR should be used with only_oneD_NMR"
    # if args['only_oneD_NMR']:
    #     assert args['combine_oneD_only_dataset'], "oneD_NMR live in both datasets"
    if args['weighted_sample_based_on_input_type']:
        assert args['combine_oneD_only_dataset'] and args['optional_inputs'], "Only available for combined dataset"
    
    seed_everything(seed=args["random_seed"])   
    
    # general args
    # apply_args(parser, args["modelname"])

    # Model args
    li_args = list(args.items())
    if args['foldername'] == "debug" or args['debug'] is True:
        args['debug'] = True
        args["epochs"] = 1
        
        

    if args['test']:
        checkpoint_path = pathlib.Path(args['checkpoint_path'])
        model_path = checkpoint_path.parents[1]

        def comment_out_problematic_yaml_lines(yaml_file_path, patterns_to_comment=None):
            """
            Comment out lines matching specific patterns in a YAML file.
            
            Args:
                yaml_file_path (str): Path to the YAML file
                patterns_to_comment (list): List of patterns to look for and comment out
                
            Returns:
                str: Path to the new YAML file
            """
            clear_stage = False
            if patterns_to_comment is None:
                patterns_to_comment = ["!!python/object/apply:pathlib.PosixPath"]
            
            with open(yaml_file_path, 'r') as f:
                lines = f.readlines()
            
            new_lines = []
            for line in lines:
                if not clear_stage:
                    if any(pattern in line for pattern in patterns_to_comment):
                        new_lines.append(f"# {line}")  # Comment out the line
                        clear_stage = True
                    else:
                        new_lines.append(line)
                else:
                    if line.startswith("- "):
                        new_lines.append(f"# {line}")
                    else:
                        new_lines.append(line)
                        clear_stage = False
            
            # new_file_path = str(yaml_file_path).replace('.yaml', '_fixed.yaml')
            with open(str(yaml_file_path), 'w') as f:
                f.writelines(new_lines)

    
        hyperpaerameters_path = model_path / "hparams.yaml"

        comment_out_problematic_yaml_lines(hyperpaerameters_path)
        with open(hyperpaerameters_path, 'r') as file:
            hparams = yaml.safe_load(file)
        args.update(hparams)
        args['test'] = True
    

    # Tensorboard setup
    
    out_path       =       DATASET_root_path / f"reproduce_previous_works/{curr_exp_folder_name}"
    # out_path =            f"/root/gurusmart/MorganFP_prediction/reproduce_previous_works/{curr_exp_folder_name}"
    out_path_final =      f"/root/gurusmart/MorganFP_prediction/reproduce_previous_works/{curr_exp_folder_name}"
    os.makedirs(out_path_final, exist_ok=True)
    os.makedirs(out_path, exist_ok=True)
    path1 = args["foldername"]
   
    path2 = args["expname"]

    # Logger setup
    my_logger = init_logger(out_path, path1, path2)
    
    my_logger.info(f'[Main] Output Path: {out_path}/{path1}/{path2}')
    try:
        my_logger.info(f'[Main] using GPU : {torch.cuda.get_device_name()}')
    except:
        my_logger.info(f'[Main] using GPU: unknown type')
    
    # FP loader setup
    if args['FP_choice'].startswith("Hash_Entropy"):
        fp_loader_configer.select_version("Hash_Entropy")
        fp_loader = fp_loader_configer.fp_loader
        radius = int(args['FP_choice'].split("_")[-1])
        fp_loader.setup(out_dim=args['out_dim'], max_radius=radius)
        
    elif args['FP_choice'].startswith("DB_specific_FP"):
        fp_loader_configer.select_version("DB_Specific")
        fp_loader = fp_loader_configer.fp_loader
        
        try:
            radius = int(args['FP_choice'].split("_")[-1])
        except ValueError:
            my_logger.info("[Main] Cannot find radius in FP_choice, using default radius 10")
            radius = 10
        fp_loader.setup(out_dim=args['out_dim'], max_radius=radius)
        if fp_loader.out_dim != args['out_dim']:
            args['out_dim'] = fp_loader.out_dim
    else:
        fp_loader_configer.select_version("MFP_Specific_Radius")
        fp_loader = fp_loader_configer.fp_loader
            
    if args['FP_choice'].startswith("pick_entropy"): # should be in the format of "pick_entropy_r9"
        only_2d = False
        FP_building_type = args['FP_building_type'].split("_")[-1]

        fp_loader.setup(only_2d=only_2d,FP_building_type=FP_building_type, out_dim=args['out_dim'])
        fp_loader.set_max_radius(int(args['FP_choice'].split("_")[-1][1:]), only_2d=only_2d)
       
    
    

    # Trainer, callbacks
    tbl = TensorBoardLogger(save_dir=out_path, name=path1, version=path2)
    metric, metricmode, patience = args["metric"], args["metricmode"], args["patience"]
    if args['optional_inputs']:
        checkpoint_callback = cb.ModelCheckpoint(monitor=f'{args["metric"].replace("/", "_")}/only_hsqc', mode=metricmode, save_top_k = 1, save_last=False)

    else:
        checkpoint_callback = cb.ModelCheckpoint(monitor=metric, mode=metricmode, save_last=False, save_top_k = 1)
        
    early_stop_metric = f'{args["metric"].replace("/", "_")}/only_hsqc' if args['optional_inputs'] else args["metric"]
    early_stopping = EarlyStopping(monitor=early_stop_metric, mode=metricmode, patience=patience)
    lr_monitor = cb.LearningRateMonitor(logging_interval="step")
    trainer = pl.Trainer(
                         max_epochs=args["epochs"],
                         accelerator="auto",
                         logger=tbl, 
                         callbacks=[early_stopping, lr_monitor, checkpoint_callback],
                        #  strategy="fsdp" if torch.cuda.device_count() > 1 else "auto",
                         accumulate_grad_batches=args["accumulate_grad_batches_num"],
                        )
    
    # Model and Data setup
    model = model_mux(parser, args["modelname"], args["load_all_weights"], args["freeze"], args, fp_loader)
    
    if trainer.global_rank == 0:
        my_logger.info(f"[Main] Model Summary: {summarize(model)}")
    data_module = data_mux(parser, args["modelname"], args["datasrc"], args["FP_choice"], args["bs"], args["ds"], args, fp_loader)
    
    
    if args["validate"]:
        my_logger.info("[Main] Just performing validation step")
        trainer.validate(model, data_module, )
    elif args['test']:
        
        trainer = pl.Trainer( accelerator="auto", accumulate_grad_batches=args["accumulate_grad_batches_num"])
        del hparams['checkpoint_path']
        hparams['test_on_deepsat_retrieval_set'] = args['test_on_deepsat_retrieval_set']
        # model = HsqcRankedTransformer.load_from_checkpoint(checkpoint_path, **hparams)     
        args = hparams

        model.setup_ranker()

        test_result = trainer.test(model, data_module, ckpt_path=checkpoint_path)

        if checkpoint_path.parts[-1].split("-")[-1].startswith("step="):
            pkl_name = "test_result.pkl"
        else:
            pkl_name = checkpoint_path.parts[-1].split("-")[-1].replace(".ckpt", ".pkl")

        if args['test_on_deepsat_retrieval_set']:
            test_result_save_parent_dir = out_path_final + "_retrieve_on_deepsat_set/" + "/".join(str(checkpoint_path).split("/")[-4:-2])
        else:
            test_result_save_parent_dir = out_path_final + "/" + "/".join(str(checkpoint_path).split("/")[-4:-2])
        os.makedirs(test_result_save_parent_dir, exist_ok=True)
            
        with open(test_result_save_parent_dir + "/" + pkl_name, "wb") as f:
            pickle.dump(test_result, f)
        
    else:
            # training
            my_logger.info("[Main] Begin Training!")
            trainer.fit(model, data_module,ckpt_path=args["checkpoint_path"])

            # Ensure all processes synchronize before switching to test mode
            trainer.strategy.barrier()

            # Now, only rank 0 will proceed to test
            if trainer.global_rank == 0:
                
                # testing
                model.setup_ranker()
                model.logger_should_sync_dist = False
                
                # my_logger.info(f"[Main] my process rank: {os.getpid()}")
                trainer = pl.Trainer( devices = 1, accumulate_grad_batches=args["accumulate_grad_batches_num"])
                my_logger.info(f"[Main] Validation metric {checkpoint_callback.monitor}, best score: {checkpoint_callback.best_model_score.item()}")
                my_logger.info(f"[Main] Testing path {checkpoint_callback.best_model_path}!")
                all_test_results = [{}]
                test_result = trainer.test(model, data_module,ckpt_path=checkpoint_callback.best_model_path)
                test_result[0]['best_epoch'] = checkpoint_callback.best_model_path.split("/")[-1].split("-")[0]
                if not args['optional_inputs']:
                
                    NP_classwise_accu = {k.split("/")[-1]:v for k,v in test_result[0].items() if "rank_1_of_NP_class" in k}
                    img_path = pathlib.Path(checkpoint_callback.best_model_path).parents[1] / f"NP_class_accu.png"
                    plot_result_dict_by_sorted_names(NP_classwise_accu, img_path)
                    all_test_results = test_result
                else:
                    # loader_all_inputs, loader_HSQC_H_NMR, loader_HSQC_C_NMR, loader_only_hsqc, loader_only_1d, loader_only_H_NMR, loader_only_C_NMR
                    for loader_idx, NMR_type in enumerate(["all_inputs", "HSQC_H_NMR", "HSQC_C_NMR", "only_hsqc", "only_1d", "only_H_NMR", "only_C_NMR"]):
                        model.only_test_this_loader(loader_idx=loader_idx)
                        # test/rank_1_of_NP_class/Sesterterpenoids/HSQC_C_NMR
                        NP_classwise_accu = {k.split("/")[-2]:v for k,v in test_result[0].items() if "rank_1_of_NP_class" in k and NMR_type in k}
                        img_path = pathlib.Path(checkpoint_callback.best_model_path).parents[1] / f"NP_class_accu_{NMR_type}.png"
                        plot_result_dict_by_sorted_names(NP_classwise_accu, img_path)
                    all_test_results = test_result
                        
                        
                # else:
                #     # loader_all_inputs, loader_HSQC_H_NMR, loader_HSQC_C_NMR, loader_only_hsqc, loader_only_1d, loader_only_H_NMR, loader_only_C_NMR
                #     data_module.setup("test")
                #     all_7_dataloaders = data_module.test_dataloader()
                #     # for loader_idx, (checkpoint_callback, curr_dataloader) in enumerate(zip(checkpoint_callbacks, all_7_dataloaders)):                               
                #     for loader_idx, curr_dataloader in enumerate(all_7_dataloaders):
                #         model.only_test_this_loader(loader_idx=loader_idx)
                        
                #         # my_logger.info(f"[Main]  monitor {monitor}!")
                #         test_result = trainer.test(model, data_module, ckpt_path=best_model_path)
                #         all_test_results[0].update(test_result[0])
                        
                #         NP_classwise_accu = {k.split("/")[-2]:v for k,v in test_result[0].items() if "rank_1_of_NP_class" in k}
                #         NMR_type = pathlib.Path(best_model_path).parts[-1].split(".")[0]
                #         img_path = pathlib.Path(best_model_path).parents[1] / f"NP_class_accu_{NMR_type}.png"
                #         plot_result_dict_by_sorted_names(NP_classwise_accu, img_path)
                        
                

                with open(f"{out_path}/{path1}/{path2}/test_result.pkl", "wb") as f:
                    pickle.dump(all_test_results, f)
                    
                
    
                my_logger.info("[Main] Done!")
                # my_logger.info("[Main] test result: \n")
                # my_logger.info(f"{test_result}")
                for key, value in all_test_results[0].items():
                    my_logger.info(f"{key}: {value}")
                if args['delete_checkpoint']:
                    os.remove(checkpoint_callback.best_model_path)
                    my_logger.info(f"[Main] Deleted checkpoint {checkpoint_callback.best_model_path}")
                os.system(f"cp -r {out_path}/* {out_path_final}/ ")
                my_logger.info(f"[Main] Copied all content from {out_path} to {out_path_final}")
                logging.shutdown()



    
