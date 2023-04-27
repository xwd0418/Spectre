import logging
from argparse import ArgumentParser

from models.ranked_transformer import HsqcRankedTransformer
from models.ranked_double_transformer import DoubleTransformer

from utils.constants import EXCLUDE_FROM_MODEL_ARGS

def model_mux(args, model_type):
  logger = logging.getLogger('logging')

  kwargs = {k: args[k] for k in args if k not in EXCLUDE_FROM_MODEL_ARGS}

  model_class = None
  if model_type == "hsqc_transformer" or model_type == "ms_transformer":
      model_class = HsqcRankedTransformer
  elif model_type == "double_transformer":
      model_class = DoubleTransformer
  else:
      raise(f"No model for model type {model_type}.")

  model = model_class(**kwargs)
  
  logger.info("[utils/init_utils/models_init.py] Freshly initializing model")
  return model