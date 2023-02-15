from datetime import datetime
from pytz import timezone

# when logging hparam locals, ignore these keys
MODEL_LOGGING_IGNORE = ["kwargs", "__class__", "self"]

# ignore when generating hyperparmaeter strings
ALWAYS_EXCLUDE = ["modelname", "debug", "expname", "foldername", "datasrc", "patience", "ds", "metric", "validate"] + \
  ["hsqc_weights", "ms_weights", "freeze", "load_all_weights"]

# priorities for hyperparameter strings
GROUPS = [
    set(["lr", "epochs", "bs"]),
    set(["heads", "layers", "hsqc_heads", "hsqc_layers", "ms_heads", "ms_layers"])
]

# things to exclude when passing var args to model
EXCLUDE_FROM_MODEL_ARGS = ["modelname", "debug", "expname", "foldername", "datasrc", "patience", "ds", "metric", \
  "freeze", "load_all_weights", "validate"]

def get_curr_time():
    pst = timezone("PST8PDT")
    california_time = datetime.now(pst)
    return california_time.strftime("%m_%d_%Y_%H:%M")