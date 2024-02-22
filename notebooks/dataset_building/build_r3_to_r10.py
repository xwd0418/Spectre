"""
building dataset with the of ranging from r3 to r10 and save
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
   
    return plain_fp

for radius in range(3, 11):
    for split in ["test", "train", "val"]:
        os.makedirs(f'/workspace/SMILES_dataset/{split}/R{radius}-6144FP/', exist_ok=True)
        
        file_names = os.listdir( f"/workspace/SMILES_dataset/{split}/HYUN_FP/")
        path_dir = f"/workspace/SMILES_dataset/{split}/"
        smiles_dict = pickle.load(open(f"{path_dir}/SMILES/index.pkl", "rb"))
        
    # %%    
        for f in tqdm.tqdm(file_names):
            idx = int(f.split(".")[0])
            smile = smiles_dict[idx]
            fp = generate_FP(smile, radius=radius)
            torch.save(fp, f'/workspace/SMILES_dataset/{split}/R{radius}-6144FP/{f}')
        print(f"R{radius} with {split} done!")        
            



