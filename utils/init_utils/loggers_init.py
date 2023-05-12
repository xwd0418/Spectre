import logging
import os
import sys
from pathlib import Path

def init_logger(out_path, path1, path2):
  logger = logging.getLogger("lightning")
  logger.setLevel(logging.DEBUG)

  logfile_dir = Path(out_path) / path1 / path2
  logfile_path = logfile_dir / "logs.txt"
  os.makedirs(logfile_dir, exist_ok=True)
  with open(logfile_path, 'w') as fp:  # touch
    pass

  formatter = logging.Formatter(
      '%(asctime)s-%(name)s-[%(filename)s:%(lineno)s]-%(message)s')
  fh = logging.FileHandler(logfile_path)
  fh.setFormatter(formatter)
  logger.addHandler(fh)
  logger.addHandler(logging.StreamHandler(sys.stdout))
  return logger
