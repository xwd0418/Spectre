import logging

from argparse import ArgumentParser

from datasets.generic_index_dataset import GenericIndexedModule
from datasets.dataset_utils import pad

from models.chemformer.utils import REGEX
from models.chemformer.tokeniser import MolEncTokeniser

def apply_args(parser: ArgumentParser):
  parser.add_argument("--data_path", type=str)
  parser.add_argument("--feats", dest="feats", action="store",
                      type=str, nargs="*", default=["HSQC", "R2-6144FP"])
  parser.add_argument("--feats_handlers", dest="feats_handlers", action="store",
                      type=str, nargs="*", default=["pad", "None"])
  parser.add_argument("--ds_path", type=str, default="tempdata/SMILES_dataset")
  parser.add_argument("--token_file", type=str,
                      default="tempdata/chemformer/bart_vocab.txt")
def map_to_handler(k, token_file):
  if k == "pad":
    return pad
  if k == "None":
    return None
  if k == "tokenise":
    MolEncTokeniser.from_vocab_file(
        token_file, REGEX, 272
    )

def data_mux(features, feature_handlers, ds_path,
             token_file=None, batch_size=32, len_override=None):
  """
      constructs data module based on model_type, and also outputs dimensions of dummy data
      (for graph visualization)
      parser: the argument parser
      len_override: to set a custom size of the dataset
  """
  if "tokenise" in feature_handlers:
    assert (token_file)

  feature_handlers = [map_to_handler(f, token_file) for f in feature_handlers]
  gim = GenericIndexedModule(ds_path, features, feature_handlers,
                             batch_size=batch_size, len_override=len_override)
  return gim
