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

  formatter_fh = logging.Formatter(
      '%(asctime)s-%(name)s-[%(filename)s:%(lineno)s]-%(message)s')
  fh = logging.FileHandler(logfile_path)
  fh.setFormatter(formatter_fh)
  logger.addHandler(fh)
  formatter_sh = logging.Formatter(
      '%(asctime)s-[%(filename)s:%(lineno)s]-%(message)s')
  sh = logging.StreamHandler(sys.stdout)
  sh.setFormatter(formatter_sh)
  logger.addHandler(sh)
  return logger
