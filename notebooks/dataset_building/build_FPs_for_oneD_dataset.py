### build entropy based FP, R2 FP, Hyun FP for oneD dataset  
"""
building dataset r2, hyun fp
"""


import time, torch, os, pickle
from fingerprint_utils import FP_generator
import tqdm
from rdkit.Chem import AllChem
from rdkit import Chem
import numpy as np


def generate_FP(smile_str, radius):
    mol = Chem.MolFromSmiles(smile_str)
    mol_H = Chem.AddHs(mol) # add implicit Hs to the molecule
    plain_fp = AllChem.GetMorganFingerprintAsBitVect(mol_H,radius=radius,nBits=6144)
   
    return torch.tensor(plain_fp).float()




dataset_to_add_FP = "OneD_Only_Dataset"
sub_dir_of_all_molecules = "oneD_NMR" # "HYUN_FP"
    
    
    
for split in ["test", "train", "val"]:
    os.makedirs(f'/workspace/{dataset_to_add_FP}/{split}/R2-6144FP/', exist_ok=True)
    os.makedirs(f'/workspace/{dataset_to_add_FP}/{split}/HYUN_FP/', exist_ok=True)
    
    file_names = os.listdir( f"/workspace/{dataset_to_add_FP}/{split}/{sub_dir_of_all_molecules}/")
    path_dir = f"/workspace/{dataset_to_add_FP}/{split}/"
    smiles_dict = pickle.load(open(f"{path_dir}/SMILES/index.pkl", "rb"))
    

    for f in tqdm.tqdm(file_names):
        idx = int(f.split(".")[0])
        smile = smiles_dict[idx]
        fp = generate_FP(smile, radius=2)
        torch.save(fp, f'/workspace/{dataset_to_add_FP}/{split}/R2-6144FP/{f}')
        
        hyun_fp = FP_generator(smile, 2)
        torch.save(torch.tensor(hyun_fp), f'/workspace/{dataset_to_add_FP}/{split}/HYUN_FP/{f}')
    print(f"{split} done!")        
            



