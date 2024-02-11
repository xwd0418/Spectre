# %%
"""
building dataset with the shortened FP and full FP from r0-r4, size 0f 30720 and reduce to 6144
"""

import time, torch, os, pickle
from fingerprint_utils import FP_generator
import tqdm
from rdkit.Chem import AllChem
from rdkit import Chem
import numpy as np

# %%
def generate_r0_to_r4_FP(smile_str):
    mol = Chem.MolFromSmiles(smile_str)
    mol_H = Chem.AddHs(mol) # add implicit Hs to the molecule
    plain_r0 = AllChem.GetMorganFingerprintAsBitVect(mol_H,radius=0,nBits=6144)
    plain_r1 = AllChem.GetMorganFingerprintAsBitVect(mol_H,radius=1,nBits=6144)
    plain_r2 = AllChem.GetMorganFingerprintAsBitVect(mol_H,radius=2,nBits=6144)
    plain_r3 = AllChem.GetMorganFingerprintAsBitVect(mol_H,radius=3,nBits=6144)
    plain_r4 = AllChem.GetMorganFingerprintAsBitVect(mol_H,radius=4,nBits=6144)
    r0_to_r4 = np.concatenate([plain_r0, plain_r1, plain_r2, plain_r3, plain_r4])
    return r0_to_r4

# %%
for split in ["test", "train", "val"]:
    os.makedirs(f'/workspace/SMILES_dataset/{split}/R0_to_R4_30720_FP/', exist_ok=True)
    os.makedirs(f'/workspace/SMILES_dataset/{split}/R0_to_R4_reduced_FP/', exist_ok=True)
    
    file_names = os.listdir( f"/workspace/SMILES_dataset/{split}/HYUN_FP/")
    path_dir = f"/workspace/SMILES_dataset/{split}/"
    smiles_dict = pickle.load(open(f"{path_dir}/SMILES/index.pkl", "rb"))
    
# %%    
    for f in tqdm.tqdm(file_names):
        idx = int(f.split(".")[0])
        smile = smiles_dict[idx]
        r0_to_r4_fp = generate_r0_to_r4_FP(smile)
        torch.save(r0_to_r4_fp, f'/workspace/SMILES_dataset/{split}/R0_to_R4_30720_FP/{f}')
        
        previously_computed_indices_to_keep = np.load("/root/MorganFP_prediction/reproduce_previous_works/smart4.5/notebooks/dataset_building/indices_kept_r0_to_r4.npy")
        torch.save(r0_to_r4_fp[previously_computed_indices_to_keep], f'/workspace/SMILES_dataset/{split}/R0_to_R4_reduced_FP/{f}')
        
        



