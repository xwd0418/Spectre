from pytorch_lightning.loggers.logger import Logger, rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_only
from collections import defaultdict
from pathlib import Path
import numpy as np
import yaml

class BestResultLogger(Logger):
  def __init__(self, save_dir, name, version):
    super().__init__()
    self.data = defaultdict(list)
    self.out_path = Path(save_dir) / name / version / "results.yml"

  @property
  def name(self):
    return "BestResultLogger"

  @property
  def version(self):
    # Return the experiment version, int or str.
    return "0.1"

  @rank_zero_only
  def log_hyperparams(self, params):
    # params is an argparse.Namespace
    # your code to record hyperparameters goes here
    pass

  @rank_zero_only
  def log_metrics(self, metrics, step):
    # metrics is a dictionary of metric names and values
    # your code to record metrics goes here
    for k, v in metrics.items():
      self.data[k].append((v, step))

  @rank_zero_only
  def save(self):
    # Optional. Any code necessary to save logger data goes here
    pass

  @rank_zero_only
  def finalize(self, status):
    # Optional. Any code that needs to be run after training
    # finishes goes here
    out_obj = {}
    for k, v in self.data.items():
      maxi_idx = np.argmax([float(i) for i, _ in v])
      maxi_v, maxi_step = v[maxi_idx]
      mini_idx = np.argmin([float(i) for i, _ in v])
      mini_v, mini_step = v[mini_idx]
      out_obj[k] = {
          "maximum": {
              "value": maxi_v,
              "iteration": maxi_step
          },
          "mini": {
              "value": mini_v,
              "iteration": mini_step
          }
      }
    with open(self.out_path, "w") as f:
      yaml.dump(out_obj, f, default_flow_style=False)
    pass
