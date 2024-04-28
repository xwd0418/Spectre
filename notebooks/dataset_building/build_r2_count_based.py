"""
building dataset, count based FP, size of 6144 , radius from 2
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
    fp =  AllChem.GetHashedMorganFingerprint(mol_H, radius=radius, nBits=6144)
    fp_dict = fp.GetNonzeroElements()
    fp_out = torch.zeros(6144)
    fp_out[list(fp_dict.keys())] = torch.tensor( list(map(lambda x:float(x), fp_dict.values())))
    return fp_out



dataset_to_add_FP = "OneD_Only_Dataset"
sub_dir_of_all_molecules = "oneD_NMR" # "HYUN_FP"


for radius in range(2,3):
    for split in ["test", "train", "val"]:
        os.makedirs(f'/workspace/{dataset_to_add_FP}/{split}/R{radius}-6144-count-based-FP/', exist_ok=True)
        
        file_names = os.listdir( f"/workspace/{dataset_to_add_FP}/{split}/{sub_dir_of_all_molecules}/")
        path_dir = f"/workspace/{dataset_to_add_FP}/{split}/"
        smiles_dict = pickle.load(open(f"{path_dir}/SMILES/index.pkl", "rb"))
        
        for f in tqdm.tqdm(file_names):
            idx = int(f.split(".")[0])
            smile = smiles_dict[idx]
            fp = generate_FP(smile, radius=radius)
            torch.save(fp, f'/workspace/{dataset_to_add_FP}/{split}/R{radius}-6144-count-based-FP/{f}')
        print(f"R{radius} with {split} done!")        
            



