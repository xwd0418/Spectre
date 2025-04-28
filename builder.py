import logging

from config import ProjectConfig
from models.ranked_transformer import HsqcRankedTransformer
from models.ranked_resnet import HsqcRankedResNet
from models.optional_input_ranked_transformer import OptionalInputRankedTransformer
from datasets.hsqc_folder_dataset import FolderDataModule
from datasets.oneD_dataset import OneDDataModule
from datasets.optional_2d_folder_dataset import OptionalInputDataModule
from utils.constants import ALWAYS_EXCLUDE, GROUPS, EXCLUDE_FROM_MODEL_ARGS, get_curr_time

def model_builder(config: ProjectConfig, fp_loader):
    base_config = config.base_config
    logger = logging.getLogger('logging')

    model_class = None
    if base_config.model_name in ("hsqc_transformer", "ms_transformer", "transformer_2d1d"):
        # TODO: verify that this is equivalent to optional_inputs = True
        if len(base_config.input_types) > 0:
            model_class = OptionalInputRankedTransformer
        else:
            model_class = HsqcRankedTransformer
    # TODO: look at names here
    elif base_config.model_name == "CNN":
        model_class = HsqcRankedResNet
    else:
        raise(f"No model for model type {base_config.model_name}.")

    if base_config.load_all_weights: # initialize with loaded state if non-empty string passed
        model = model_class.load_from_checkpoint(base_config.load_all_weights, strict=False)
        logger.info("[Main] Loading model from weights")
    else: # or from scratch
        print(base_config.model_name)
        model = model_class(config, fp_loader)
        logger.info("[Main] Freshly initializing model")

    if base_config.freeze:
        logger.info("[Main] Freezing model weights")
        for param in model.parameters():
            param.requires_grad = False
    return model

def data_builder(config: ProjectConfig, fp_loader):
    """
        constructs data module based on model_type, and also outputs dimensions of dummy data
        (for graph visualization)
    """
    base_config = config.base_config
    batch_size = base_config.batch_size
    kwargs = args # vars(parser.parse_args())

    # TODO: verify that this is equivalent to optional_inputs = True
    if len(base_config.input_types) > 0:
        return OptionalInputDataModule(dir=base_config.data_folder, FP_choice=FP_choice, input_src=["HSQC", "oneD_NMR"], fp_loader=fp_loader, batch_size=batch_size, parser_args=kwargs)
    # OneD datamodule is somewhat buggy, so we will not use hsqc_folder_dataset.py 
    if args['only_oneD_NMR']:
        # oned_dir = "/workspace/OneD_Only_Dataset"
        # here the choice is still SMILES_dataset, but infact, OneDDataset use both datasets
        return OneDDataModule(dir=base_config.data_folder, FP_choice=FP_choice, fp_loader=fp_loader, batch_size=batch_size, parser_args=kwargs)
    elif base_config.model_name == "hsqc_transformer":
        return FolderDataModule(dir=base_config.data_folder, FP_choice=FP_choice, input_src=["HSQC"], fp_loader=fp_loader, batch_size=batch_size, parser_args=kwargs)
    elif base_config.model_name == "CNN":
        num_channels = kwargs['num_input_channels']
        return FolderDataModule(dir=base_config.data_folder, FP_choice=FP_choice, input_src=[f"HSQC_images_{num_channels}channel"], fp_loader=fp_loader, batch_size=batch_size, parser_args=kwargs)
    elif base_config.model_name == "ms_transformer":
        return FolderDataModule(dir=base_config.data_folder, FP_choice=FP_choice, input_src=["MS"], fp_loader=fp_loader, batch_size=batch_size, parser_args=kwargs)
    elif base_config.model_name == "transformer_2d1d":
        if kwargs['use_oneD_NMR_no_solvent']:
            return FolderDataModule(dir=base_config.data_folder, FP_choice=FP_choice, input_src=["HSQC", "oneD_NMR"], fp_loader=fp_loader, batch_size=batch_size, parser_args=kwargs)
        else:
            return FolderDataModule(dir=base_config.data_folder, FP_choice=FP_choice, input_src=["HSQC"], fp_loader=fp_loader, batch_size=batch_size, parser_args=kwargs)
    
    raise(f"No datamodule for model type {base_config.model_name}.")