'''This script adds the 1D NMR data to the cleaned dataset.'''

import os, torch, pickle, json, random
import sys
from rdkit import Chem
from tqdm import tqdm
from collections import defaultdict

from bs4 import BeautifulSoup
import  requests


'''This function reads the 1D NMR data from a file and returns the carbon and hydrogen tensors.'''
def get_nmr_tensors(file_path):
    c_values = set()
    h_values = set()

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            if len(parts) < 3 or not parts[2]:
                continue  # Skip lines that don't have enough parts or the third column is empty

            element, _, value, *_ = parts
            value = float(value)

            if element == 'C':
                c_values.add(value)
            elif element == 'H':
                h_values.add(value)

    # Convert sets to tensors
    c_tensor = torch.tensor(list(c_values), dtype=torch.float32)
    h_tensor = torch.tensor(list(h_values), dtype=torch.float32)
    
    return c_tensor, h_tensor

# Load the NP-MRD files and the mapping from NP to InChI
saved_data=pickle.load(open(f"/root/gurusmart/MorganFP_prediction/task_scripts/inchi_mapping.pkl", 'rb'))
NP_to_inchi = saved_data['NP_to_inchi']
inchi_to_file_path = saved_data['inchi_to_file_path']

for split in ["test", "train", "val"]:
    os.makedirs(f'/workspace/SMILES_dataset/{split}/detailed_oneD_NMR/', exist_ok=True)

NP_MRD_FILES_dir = '/root/gurusmart/data/NP-MRD-dataset/NP-MRD-shift-assignments'
npmrd_files_txt_only = sorted(os.listdir(NP_MRD_FILES_dir))



'''This function extracts the solvent from the file name, 
by sending a GET request to the NP-MRD website and parsing the title tag of the response.'''
def get_solvent_from_file_name(f):
    spectrum_id = f.split("_")[2]
    url = f'https://np-mrd.org/spectra/nmr_one_d/{spectrum_id}' #1113
    # Send a GET request
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    title_tag = soup.find('title')
    return title_tag.text.split(':')[-1].strip().split(",")[2].strip() if title_tag else "unknown"
    
# get_solvent_from_file_name('NP0000007_nmroned_26098_2774638.txt')

# Add the 1D NMR data to the cleaned dataset
added_inchis = set()
for iter, f in enumerate(npmrd_files_txt_only):
    if iter % 500 == 0:
        # print(iter, len(npmrd_files_txt_only))
        print(f"Progress: {iter}/{len(npmrd_files_txt_only)}")
    if f.split("_")[0] not in NP_to_inchi: #no inchi npmrd case
        continue
    inchi =  NP_to_inchi[f.split("_")[0]]
    if inchi in added_inchis: # do not repeat work
        continue 
    if inchi:
        if inchi not in inchi_to_file_path: # no same compound in smart dataset
            continue
        c_tensor, h_tensor = get_nmr_tensors(os.path.join(NP_MRD_FILES_dir, f))
        if len(c_tensor) == 0 and len(h_tensor)==0:
            continue
        try:
            solvent = get_solvent_from_file_name(f)
        except:
            solvent = "unknown"
        file_path = inchi_to_file_path[inchi][-1]
        if solvent == "unknown" and os.path.exists('/workspace/SMILES_dataset/'+file_path):
            continue
        torch.save([c_tensor, h_tensor, solvent], '/workspace/SMILES_dataset/'+file_path )
        if solvent != "unknown":
            added_inchis.add(inchi)
    
    





