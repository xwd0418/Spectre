import logging
import os
import sys
import torch
import pytorch_lightning as pl
import pytorch_lightning.callbacks as cb
from utils.init_utils import args_init, data_init, loggers_init, models_init
from models.extras.best_results_logger import BestResultLogger

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from argparse import ArgumentParser

def main():
  parser = ArgumentParser(add_help=True)
  args_init.training_args(parser)
  args = vars(parser.parse_known_args()[0])

  # general args
  args_init.apply_args(parser, args["modelname"])

  # Model args
  args_with_model = vars(parser.parse_known_args()[0])
  li_args = list(args_with_model.items())

  # Tensorboard setup
  out_path = "/data/smart4.5"
  path1 = args["foldername"]  # lightning_logs
  path2 = args["expname"]

  # Logger setup
  my_logger = loggers_init.init_logger(out_path, path1, path2)
  my_logger.info(f'[Main - Logger] Output Path: {out_path}/{path1}/{path2}')

  # Model and Data setup
  model = models_init.model_mux(parser, args["modelname"])
  batch_size = args["batch_size"]
  data_module = data_init.data_mux(
      parser, len_override=args["data_len"], batch_size=batch_size)
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

  # Create trainer instance
  trainer = pl.Trainer(max_epochs=args["epochs"], gpus=1, logger=[tbl, brl], callbacks=[
                       checkpoint_callback, early_stopping, lr_monitor])

  my_logger.info("[Main] Begin Training!")
  trainer.fit(model, data_module)
  my_logger.info("[Main] Done Training!")


if __name__ == '__main__':
  main()
