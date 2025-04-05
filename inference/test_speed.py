import time

import sys, os
sys.path.insert(0,"/root/gurusmart/MorganFP_prediction/reproduce_previous_works/Spectre")
from inference.inference_utils import choose_model

print(1)

# load model 
from datasets.dataset_utils import fp_loader_configer

fp_loader_configer.select_version("Hash_Entropy")

print(2)
start_time = time.time()

hparams, model  = choose_model("C-NMR")


end_time = time.time()
print(f"Model loaded in {end_time - start_time:.2f} seconds")
