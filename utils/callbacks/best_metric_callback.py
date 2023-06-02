from copy import deepcopy
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from utils.init_utils.loggers_init import get_logger

class BestMetricCallback(Callback):
  def __init__(self, metrics, metric_modes):
    self.metrics = metrics
    self.metric_modes = metric_modes
    assert(len(metrics) = len(metric_modes))
    self.logger = get_logger()

  def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
    callbacks = deepcopy(trainer.callback_metrics)
    for metric in self.metrics:
      if metric not in callbacks:
        raise Exception(f"Metric {metric} not in callbacks") 
    self.logger.info(f"BMC, {}")
    pass
