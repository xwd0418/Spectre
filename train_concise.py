import logging
import os
import sys
import torch
import pytorch_lightning as pl
import pytorch_lightning.callbacks as cb
from utils.init_utils import (
    args_init, data_init, loggers_init, models_init
)
from utils import marker, config
from models.extras.best_results_logger import BestResultLogger

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from pathlib import Path
from argparse import ArgumentParser

def main():
  parser = ArgumentParser(add_help=True)
  args_init.training_args(parser)
  args = vars(parser.parse_known_args()[0])
  is_config = False
  if args["config"]:
    args = config.load_single_config(args["config"])
    is_config = True

  # Tensorboard setup
  out_path = "/data/smart4.5"
  path1 = args["foldername"]  # lightning_logs
  path2 = args["expname"]
  marker_path = Path(out_path) / path1 / path2 / "marker"

  # Logger setup
  my_logger = loggers_init.init_logger(out_path, path1, path2)
  my_logger.info(f'[Main - Logger] Output Path: {out_path}/{path1}/{path2}')

  # marking experiment as done
  if not args["force_start"] and marker.has_marker(marker_path):
    print("Experiment in progress / done")
    exit(123)
  os.makedirs(Path(out_path) / path1 / path2, exist_ok=True)
  marker.place_marker(marker_path)

  if not is_config:
    # general args
    args_init.apply_args(parser, args["modelname"])
    # Model args
    data_init.apply_args(parser)
    args = vars(parser.parse_known_args())

  # Model and Data setup
  batch_size = args["batch_size"]
  model = models_init.model_mux(args, args["modelname"])
  data_module = data_init.data_mux(
      args["feats"], args["feats_handlers"], args["ds_path"],
      token_file=args.get("token_file"),
      len_override=args.get("data_len"), batch_size=batch_size)
  my_logger.info(f"[Main - Data] Initialized.")

  # All callbacks
  metric, metricmode, patience = args["metric"], args["metricmode"], args["patience"]

  brl = BestResultLogger(save_dir=out_path, name=path1, version=path2)
  tbl = TensorBoardLogger(save_dir=out_path, name=path1, version=path2)
  checkpoint_callback = cb.ModelCheckpoint(
      monitor=metric, mode=metricmode, save_last=True, save_top_k=3)
  early_stopping = EarlyStopping(
      monitor=metric, mode=metricmode, patience=patience)
  lr_monitor = cb.LearningRateMonitor(logging_interval="step")

  if args.get("actually_run", True):
    # Create trainer instance
    trainer = pl.Trainer(max_epochs=args["epochs"], accelerator="gpu", devices=1, logger=[tbl, brl], callbacks=[
        checkpoint_callback, early_stopping, lr_monitor])

    my_logger.info("[Main] Begin Training!")
    trainer.fit(model, data_module)
    my_logger.info("[Main] Done Training!")


if __name__ == '__main__':
  main()
