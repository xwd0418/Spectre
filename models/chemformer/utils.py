import torch

# Default model hyperparams
DEFAULT_D_MODEL = 512
DEFAULT_NUM_LAYERS = 6
DEFAULT_NUM_HEADS = 8
DEFAULT_D_FEEDFORWARD = 2048
DEFAULT_ACTIVATION = "gelu"
DEFAULT_MAX_SEQ_LEN = 512
DEFAULT_DROPOUT = 0.1

DEFAULT_DEEPSPEED_CONFIG_PATH = "ds_config.json"
DEFAULT_LOG_DIR = "tb_logs"
DEFAULT_VOCAB_PATH = "bart_vocab.txt"
DEFAULT_CHEM_TOKEN_START = 272
REGEX = "\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9]"

DEFAULT_GPUS = 1
DEFAULT_NUM_NODES = 1

USE_GPU = True
use_gpu = USE_GPU and torch.cuda.is_available()
