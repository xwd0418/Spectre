from argparse import ArgumentParser

from models.ranked_transformer import HsqcRankedTransformer
from models.ranked_double_transformer import DoubleTransformer

def apply_args(parser, model_type):
  if model_type == "hsqc_transformer" or model_type == "ms_transformer":
    HsqcRankedTransformer.add_model_specific_args(parser)
  elif model_type == "double_transformer":
    DoubleTransformer.add_model_specific_args(parser)
  else:
    raise (f"No model for model type {model_type}.")

def training_args(parser):
  # string that is basically a constant
  parser.add_argument("--config", type=str, default=None,
                      help="config file, or empty to use cli args")
  parser.add_argument("--modelname", type=str)

  parser.add_argument("--epochs", type=int, default=120)
  # Force the experiment to start (ignore marker checking)
  parser.add_argument("--force_start", type=bool, default=False)
  parser.add_argument("--actually_run", type=bool, default=True)

  # logging args
  parser.add_argument("--foldername", type=str, default=f"lightning_logs")
  parser.add_argument("--expname", type=str, default=f"experiment")

  parser.add_argument("--bs", type=int, default=64)
  parser.add_argument("--patience", type=int, default=30)

  # for early stopping/model saving
  parser.add_argument("--metric", type=str, default="val/mean_ce_loss")
  parser.add_argument("--metricmode", type=str, default="max")

  # data args
  parser.add_argument("--data_len", type=int, default=None)
  parser.add_argument("--batch_size", type=int, default=32)
