from copy import deepcopy
from typing import Any, Dict
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from utils.init_utils.loggers_init import get_logger

class BestMetricCallback(Callback):
  def __init__(self, metrics, metric_modes):
    self.metrics = metrics
    self.metric_modes = metric_modes
    assert(len(metrics) == len(metric_modes))
    self.state = {}
    for k in metrics:
      self.state[k] = None
    self.logger = get_logger()

  def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
    self.logger.info("load_state_dict")
    self.state = state_dict
    
  def state_dict(self) -> Dict[str, Any]:
    self.logger.info("state_dict")
    return self.state
  
  def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
    self.logger.info("train end")
    
  
  def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
    callbacks = deepcopy(trainer.callback_metrics)
    for metric in self.metrics:
      if metric not in callbacks:
        raise Exception(f"Metric {metric} not in callbacks")

    self.logger.info(f"BMC, {}")
    pass
