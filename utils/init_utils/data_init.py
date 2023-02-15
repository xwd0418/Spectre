import logging

from datasets.generic_index_dataset import GenericIndexedModule

def data_mux(parser, len_override = None):
  """
      constructs data module based on model_type, and also outputs dimensions of dummy data
      (for graph visualization)
      parser: the argument parser
      len_override: to set a custom size of the dataset
  """
  logger = logging.getLogger("lightning")

  SMILES_dataset_path = "tempdata/SMILES_dataset"
  features = ["HSQC", "HYUN_FP"]
  feature_handlers = [pad, None]
  gim = GenericIndexedModule(direct, features, feature_handlers, 
    batch_size = batch_size, len_override = len_override)
  return gmi