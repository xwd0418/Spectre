import os
import pathlib
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message="You are using `torch.load` with `weights_only=False`")
warnings.filterwarnings("ignore", category=UserWarning, message="The PyTorch API of nested tensors is in prototype stage and will change in the near future.")

from omegaconf import OmegaConf
import hydra
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning.callbacks as cb
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.utilities.model_summary import summarize

from builder import model_builder, data_builder
from config import ProjectConfig
from util import seed_everything, init_logger, get_fp_loader
    
def train(config: ProjectConfig):
    pass

def val(config: ProjectConfig):
    pass

def test(config: ProjectConfig):
    pass

@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg):
    config = ProjectConfig(config = OmegaConf.to_object(cfg))
    base_config, model_args = config.base_config, config.model_args
    print(config)
    torch.set_float32_matmul_precision('medium')
    
    #if args['weighted_sample_based_on_input_type']:
    #    assert args['combine_oneD_only_dataset'] and args['optional_inputs'], "Only available for combined dataset"
    # TODO: validation
    if base_config.weighted_sample_based_on_input_type:
        pass
    
    seed_everything(base_config.random_seed)
    
    # TODO: do we need this?
    if base_config.log_folder == 'debug' or base_config.debug:
        base_config.debug = True
        base_config.epochs = 1
    
    # TODO: fix hyperparam update
    if base_config.mode ==  'test':
        checkpoint_path = pathlib.Path(base_config.checkpoint_path)
        model_path = checkpoint_path.parents[1] # TODO: check what this is doing
        hyperparameter_path = model_path / "hparams.yaml"
        # with open(hyperparameter_path, 'r') as file:
        #    hparams = yaml.safe_load(file)
        # args.update(hparams) TODO: update params
    
    # TODO: why is out_path at /workspace and why /reproduce_previous_works
    out_path = os.path.join(base_config.data_folder, f"reproduce_previous_works/{base_config.experiment_name}")
    out_path_final = f"/root/gurusmart/MorganFP_prediction/reproduce_previous_works/{base_config.experiment_name}"
    os.makedirs(out_path_final, exist_ok=True)
    os.makedirs(out_path, exist_ok=True)

    # TODO: fix log path
    logger_path = os.path.join(out_path, base_config.log_folder, base_config.experiment_name)
    my_logger = init_logger(logger_path)
    
    my_logger.info(f'[Main] Logger path: {logger_path}')
    try:
        my_logger.info(f'[Main] Using GPU: {torch.cuda.get_device_name()}')
    except:
        my_logger.info(f'[Main] Using GPU: Unknown type')
    
    # FP Loader
    # TODO: fp naming
    # TODO: when fp choice is mfp_specific_radius, there is something about pick_entropy. what is the point
    fp_loader = get_fp_loader(base_config)
    
    # Trainer, callbacks
    tbl = TensorBoardLogger(save_dir=out_path, name=base_config.log_folder, version=base_config.experiment_name)
    metric, metricmode, patience = base_config.metric, base_config.metricmode, base_config.patience
    # TODO: why did original code replace "/" with "_" if optional_inputs = True but not otherwise?
    # TODO: verify that this is equivalent to optional_inputs = True
    if len(base_config.input_types) > 0:
        checkpoint_callback = cb.ModelCheckpoint(
            monitor = f'{metric.replace("/", "_")}/only_hsqc',
            mode = metricmode,
            save_top_k = 1,
            save_last = False
        )
        early_stop_metric = f'{metric.replace("/", "_")}/only_hsqc'
    else:
        checkpoint_callback = cb.ModelCheckpoint(
            monitor = metric,
            mode = metricmode,
            save_last = False,
            save_top_k = 1
        )
        early_stop_metric = metric
    early_stopping = EarlyStopping(monitor=early_stop_metric, mode=metricmode, patience=patience)
    lr_monitor = cb.LearningRateMonitor(logging_interval="step")
    trainer = pl.Trainer(
        max_epochs=base_config.epochs,
        accelerator="auto",
        logger=tbl, 
        callbacks=[early_stopping, lr_monitor, checkpoint_callback],
        accumulate_grad_batches=base_config.accumulate_grad_batches_num,
    )
    
    # Model and Data setup
    model = model_builder(config, fp_loader)
    
    if trainer.global_rank == 0:
        my_logger.info(f"[Main] Model Summary: {summarize(model)}")
        
    return
    data_module = data_builder(config, fp_loader)
       
    if config.base_config.mode == 'train':
        pass
    elif config.base_config.mode == 'val':
        pass
    elif config.base_config.mode == 'test':
        pass
    
if __name__ == "__main__":
    main()