import logging
import os
from argparse import ArgumentParser

from datasets.generic_index_dataset import GenericIndexedModule
from datasets.dataset_utils import pad, tokenise_and_mask

from pysmilesutils.augment import SMILESAugmenter
from models.chemformer.utils import REGEX
from models.chemformer.tokeniser import MolEncTokeniser

def apply_args(parser: ArgumentParser):
  parser.add_argument("--feats", action="store",
                      type=str, nargs="*", default=["HSQC", "R2-6144FP"])
  parser.add_argument("--feats_handlers", action="store",
                      type=str, nargs="*", default=["pad", "None"])
  parser.add_argument("--ds_path", type=str, default="tempdata/SMILES_dataset")
  parser.add_argument("--token_file", type=str,
                      default="tempdata/chemformer/bart_vocab.txt")
  parser.add_argument("--num_workers", type=int, default=4)
def map_to_handler(k, token_file):
  if k == "pad":
    return pad
  if k == "None":
    return None
  if k == "tokenise":
    tokeniser = MolEncTokeniser.from_vocab_file(
        token_file, REGEX, 272
    )
    aug = SMILESAugmenter()  # random enumeration

    def tokenise_fn(smiles):
      return tokenise_and_mask(aug(smiles), tokeniser)
    return tokenise_fn

def data_mux(features, feature_handlers, ds_path,
             token_file=None, batch_size=32, len_override=None,
             num_workers=4):
  """
      constructs data module based on model_type, and also outputs dimensions of dummy data
      (for graph visualization)
      parser: the argument parser
      len_override: to set a custom size of the dataset
  """
  logger = logging.getLogger('lightning')

  if "tokenise" in feature_handlers:
    assert (token_file)

  feature_handlers = [map_to_handler(f, token_file) for f in feature_handlers]
  gim = GenericIndexedModule(ds_path, features, feature_handlers,
                             batch_size=batch_size, len_override=len_override,
                             num_workers=num_workers)
  logger.info(f"[data_init] Attempting to use {num_workers} cpus")
  return gim
