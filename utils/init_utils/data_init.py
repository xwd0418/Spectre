import logging

from datasets.generic_index_dataset import GenericIndexedModule
from datasets.dataset_utils import pad

def data_mux(parser, len_override = None, batch_size = 32):
  """
      constructs data module based on model_type, and also outputs dimensions of dummy data
      (for graph visualization)
      parser: the argument parser
      len_override: to set a custom size of the dataset
  """
  logger = logging.getLogger("lightning")

  SMILES_dataset_path = "tempdata/SMILES_dataset"
  features = ["HSQC", "R2-6144FP"]
  feature_handlers = [pad, None]
  gim = GenericIndexedModule(SMILES_dataset_path, features, feature_handlers, 
    batch_size = batch_size, len_override = len_override)
  return gim