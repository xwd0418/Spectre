# extract NP_MRD shift assginemnt txt files 

import zipfile
import os
import tqdm

zip_path = "/root/gurusmart/data/shift_assignment.zip"
extract_path = "/root/gurusmart/data/NP-MRD-dataset/NP-MRD-shift-assignments"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    extracted_prefixes = set()
    for file in (zip_ref.namelist()):
        prefix = file.split('_')[0]
        suffix = file.split('.')[-1]
        if prefix not in extracted_prefixes and suffix=="txt" and "twod" not in file:
            zip_ref.extract(file, extract_path)
            extracted_prefixes.add(prefix)
    
