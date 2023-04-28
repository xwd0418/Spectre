import logging
import pytorch_lightning as pl
from argparse import ArgumentParser

from models.ranked_transformer import HsqcRankedTransformer, Moonshot
from models.ranked_double_transformer import DoubleTransformer


from utils.constants import EXCLUDE_FROM_MODEL_ARGS

def model_mux(args, model_type):
  logger = logging.getLogger('lightning')

  kwargs = {k: args[k] for k in args if k not in EXCLUDE_FROM_MODEL_ARGS}
  load_override = {} if "load_override" not in args else args["load_override"]

  model_class: pl.LightningModule = None
  if model_type == "hsqc_transformer" or model_type == "ms_transformer":
    model_class = HsqcRankedTransformer
  elif model_type == "double_transformer":
    model_class = DoubleTransformer
  elif model_type == "moonshot":
    model_class = Moonshot
  else:
    raise (f"No model for model type {model_type}.")

  if not args.get("load_path"):
    logger.info("[models_init.py] Freshly initializing model")
    model = model_class(**kwargs)
  else:
    logger.info("[models_init.py] Loading from checkpoint")
    model = model_class.load_from_checkpoint(
        args.get("load_path"),
        strict=False,
        **load_override
    )
  return model
