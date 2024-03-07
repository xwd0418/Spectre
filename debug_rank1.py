import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from models.ranked_transformer import HsqcRankedTransformer
from datasets.hsqc_folder_dataset import FolderDataModule
import yaml
from pytorch_lightning.loggers import TensorBoardLogger
torch.set_float32_matmul_precision('medium')

# Load the checkpoint from the path
checkpoint_path = \
"/root/MorganFP_prediction/reproduce_previous_works/compute_ranking_lower_bound/count_based_FP/count_based_CE/checkpoints/epoch=34-step=59990.ckpt"



hyperpaerameters_path = checkpoint_path.split("checkpoints")[0] + "hparams.yaml"
# # Load the YAML file
with open(hyperpaerameters_path, 'r') as file:
    hparams = yaml.safe_load(file)

# hparams["FP_choice"]='R2-6144FP'
# hparams["warm_up_steps"] = 10000

model = HsqcRankedTransformer.load_from_checkpoint(checkpoint_path)
model.change_ranker_for_testing()

data_module = FolderDataModule(dir="/workspace/SMILES_dataset", 
                               input_src=["HSQC", "oneD_NMR"], 
                               FP_choice=hparams["FP_choice"],
                               batch_size=64, parser_args=hparams)
# 1363

tbl = TensorBoardLogger(save_dir="/root/MorganFP_prediction/reproduce_previous_works/test_results", 
                        name="test_sample", 
                        version="check_saving_ckpt")

# Create a trainer instance
trainer = Trainer(logger=tbl )
# Test the model
test_result = trainer.test(model, datamodule=data_module)

# # Save the test results to the test folder
# test_results_path = "/root/MorganFP_prediction/reproduce_previous_works/test_results"
# torch.save(trainer.test_dataloaders[0].dataset, test_results_path)