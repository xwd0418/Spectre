from argparse import ArgumentParser
from pathlib import Path

import pytorch_lightning as pl
import pytorch_lightning.callbacks as cb
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.tuner import Tuner

from utils import config
from utils.callbacks.best_metric_callback import BestMetricCallback
from utils.callbacks.marker_callback import DoneMarkerCallback
from utils.init_utils import (args_init, data_init, loggers_init, models_init,
                              warnings_init)
from utils.loggers.betterTBL import BetterTBL


def main():
  parser = ArgumentParser(add_help=True)
  args_init.training_args(parser)
  args = vars(parser.parse_known_args()[0])
  is_config = False
  if args["config"]:
    args = config.load_single_config(args["config"])
    is_config = True

  # Tensorboard setup
  out_path = args.get("out_path", "/data/smart4.5")
  path1 = args["foldername"]  # lightning_logs
  path2 = args["expname"]

  # Logger setup
  warnings_init.clear_warnings()
  my_logger = loggers_init.init_logger(out_path, path1, path2)
  my_logger.info(f'[Main - Logger] Output Path: {out_path}/{path1}/{path2}')

  # seed
  if args.get("seed"):
    pl.utilities.seed.isolate_rng(args.get("seed"))
    my_logger.info(f"[Main] Using seed {args.get('seed')}")

  if not is_config:
    # general args
    args_init.apply_args(parser, args["modelname"])
    # Model args
    data_init.apply_args(parser)
    args = vars(parser.parse_known_args())

  # Model and Data setup
  batch_size = args["batch_size"]
  model = models_init.model_mux(args, args["modelname"])
  data_module_type = args["module_type"]
  data_module = data_init.data_mux(
      data_module_type,
      features=args.get("feats"), feature_handlers=args.get("feats_handlers"), ds_path=args.get("ds_path"),
      token_file=args.get("token_file"), len_override=args.get("data_len"), batch_size=batch_size, num_workers=args["num_workers"])
  my_logger.info(f"[Main - Data] Initialized.")

  # All callbacks
  metric, metricmode, patience = args["metric"], args["metricmode"], args["patience"]

  tbl = TensorBoardLogger(save_dir=out_path,
                          name=path1, version=path2, default_hp_metric=False)
  checkpoint_callback = cb.ModelCheckpoint(
      monitor=metric, mode=metricmode, save_last=True, save_top_k=3)
  early_stopping = EarlyStopping(
      monitor=metric, mode=metricmode, patience=patience)
  lr_monitor = cb.LearningRateMonitor(logging_interval="step")
  best_metrics, best_metric_modes = args.get(
      "best_metrics", []), args.get("best_metric_modes", [])
  best_metric_callback = BestMetricCallback(
      best_metrics, best_metric_modes)

  if args.get("actually_run", True):
    # Create trainer instance
    trainer = pl.Trainer(max_epochs=args["epochs"], accelerator="gpu",
                         devices=1, logger=[tbl], callbacks=[
        checkpoint_callback, early_stopping, lr_monitor, best_metric_callback
    ])

    my_logger.info("[Main] Begin Training!")
    trainer.fit(model, datamodule=data_module, ckpt_path="last")
    my_logger.info("[Main] Done Training!")


if __name__ == '__main__':
  main()
