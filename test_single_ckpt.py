import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from models.ranked_transformer import HsqcRankedTransformer
from datasets.hsqc_folder_dataset import FolderDataModule
import yaml
from pytorch_lightning.loggers import TensorBoardLogger

# Load the checkpoint from the path
checkpoint_path = "/root/MorganFP_prediction/reproduce_previous_works/reproduce_w_cleaned_dataset_n_testing_ranker/embedding/everything/checkpoints/last.ckpt"
hyperpaerameters_path = checkpoint_path.split("checkpoints")[0] + "hparams.yaml"
# Load the YAML file
with open(hyperpaerameters_path, 'r') as file:
    hparams = yaml.safe_load(file)
if "normalize_hsqc" not in hparams:
    hparams["normalize_hsqc"] = False
if "disable_hsqc_peaks" not in hparams:
    hparams["disable_hsqc_peaks"] = False
if "enable_hsqc_delimeter_only_2d" not in hparams:
    hparams["enable_hsqc_delimeter_only_2d"] = False
if "do_hyun_fp" not in hparams:
    hparams["do_hyun_fp"] = False
if "num_workers" not in hparams:
    hparams["num_workers"] = 12
if "disable_solvent" not in hparams:
    hparams["disable_solvent"] = False
    
model = HsqcRankedTransformer.load_from_checkpoint(checkpoint_path)
model.change_ranker_for_testing()

data_module = FolderDataModule(dir="/workspace/SMILES_dataset", 
                               do_hyun_fp=False,
                               input_src=["HSQC", "detailed_oneD_NMR"], 
                               batch_size=64, parser_args=hparams)

tbl = TensorBoardLogger(save_dir="/root/MorganFP_prediction/reproduce_previous_works/test_results", 
                        name="test_sample", 
                        version="check_saving_ckpt")

# Create a trainer instance
trainer = Trainer(devices=1,logger=tbl )
# Test the model
trainer.test(model, datamodule=data_module)

# # Save the test results to the test folder
# test_results_path = "/root/MorganFP_prediction/reproduce_previous_works/test_results"
# torch.save(trainer.test_dataloaders[0].dataset, test_results_path)