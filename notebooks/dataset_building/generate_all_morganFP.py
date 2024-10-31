# %% [markdown]
# 2024/4/19
# 
# build r0-r'n' Fingerprint with exact FP and normal FP
# with n in [1,15]
# 
# In the end, we will (usually) pick the 6144 positions with the highest entropy   
# 

# %%

import pickle
from pathlib import Path
import time, torch, os
from fingerprint_utils import FP_generator
batch_size=64
import tqdm
import numpy as np
from matplotlib import pyplot as plt
from rdkit.Chem import AllChem
from rdkit import Chem
import sys, pathlib
repo_path = pathlib.Path(__file__).resolve().parents[2]

# %%
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

def generate_FP_with_exact_radius(mol, radius=2, length=6144):

    # Dictionary to store information about which substructures contribute to setting which bits
    bitInfo = {}
    
    # Generate the fingerprint with bitInfo to track the substructures contributing to each bit
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=length, bitInfo=bitInfo)
    
    # Create an array of zeros to represent the new fingerprint
    new_fp = [0] * length
    
    # Filter bitInfo to keep only entries where substructures have the exact radius
    for bit, atoms in bitInfo.items():
        # Check if any substructure at this bit has the exact specified radius
        if any(radius_tuple[1] == radius for radius_tuple in atoms):
            # Set the corresponding bit in the new fingerprint
            new_fp[bit] = 1
    
    # Return the new filtered fingerprint as a list of bits
    return new_fp


def generate_normal_FP(mol, radius=2, length=6144):

    # Dictionary to store information about which substructures contribute to setting which bits
    bitInfo = {}
    
    # Generate the fingerprint with bitInfo to track the substructures contributing to each bit
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=length, bitInfo=bitInfo)
    new_fp = [0] * length
    for bit, atoms in bitInfo.items():
            new_fp[bit] = 1
            
    return new_fp
   


# %%

def generate_FP_on_bits_with_exact_radius(mol, radius=2, length=6144):

    # Dictionary to store information about which substructures contribute to setting which bits
    bitInfo = {}
    on_bits = []
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=length, bitInfo=bitInfo)
    # Filter bitInfo to keep only entries where substructures have the exact radius
    for bit, atoms in bitInfo.items():
        # Check if any substructure at this bit has the exact specified radius
        if any(radius_tuple[1] == radius for radius_tuple in atoms):
            # Set the corresponding bit in the new fingerprint
            on_bits.append(bit)
    
    # Return the new filtered fingerprint as a list of bits
    return np.array(on_bits)


def generate_normal_FP_on_bits(mol, radius=2, length=6144):

    # Dictionary to store information about which substructures contribute to setting which bits
    bitInfo = {}
    
    # Generate the fingerprint with bitInfo to track the substructures contributing to each bit
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=length, bitInfo=bitInfo)
    on_bits = np.array(fp.GetOnBits())
    return on_bits

# %%
'''This generates exact-r FP and save the count at each position !!!
Gonna take you a few minutes 
'''

os.makedirs(Path(f'{repo_path}/notebooks/dataset_building/FP_on_bits_pickles'), exist_ok=True)
def generate_FP_indices_of_r0_r15(split, FP_length, generation_method, dataset="2d"):
    num_plain_FPs = 16
    if generation_method == "exact":
        generate_FP_on_bis = generate_FP_on_bits_with_exact_radius
        save_name = f"Exact_FP_on_bits_r0_r15_len_{FP_length}_{dataset}_{split}.pkl"
    elif generation_method == "normal":
        generate_FP_on_bis = generate_normal_FP_on_bits
        save_name = f"Normal_FP_on_bits_r0_r15_len_{FP_length}_{dataset}_{split}.pkl"
    else:
        raise ValueError("generation_method should be exact or normal")
    if dataset=="2d":
        path_dir = Path("/workspace/SMILES_dataset/")
    elif dataset=="1d":
        path_dir = Path("/workspace/OneD_Only_Dataset/")
    else:
        raise ValueError("dataset should be 2d or 1d")    
    smile_nmr = pickle.load(open(path_dir / split/ "SMILES/index.pkl", "rb"))

    FP_on_bits = {}
    for file_idx, smile_str in tqdm.tqdm(smile_nmr.items()):
        mol = Chem.MolFromSmiles(smile_str)
        mol_H = Chem.AddHs(mol) # add implicit Hs to the molecule
        all_plain_fps = []
        for radius in range(num_plain_FPs):
            all_plain_fps.append(generate_FP_on_bis(mol_H, radius=radius, length=FP_length) + radius*FP_length)
        concated_FP = np.concatenate(all_plain_fps)

        FP_on_bits[file_idx] = concated_FP

    save_dir = Path(f'{repo_path}/notebooks/dataset_building/FP_on_bits_pickles')
    
    FP_on_bits_path = save_dir / save_name 
    with open(FP_on_bits_path, 'wb') as f:
        pickle.dump(FP_on_bits, f)
        
    return FP_on_bits

# count = np.zeros(6144*num_plain_FPs)
    
    
    
    # count+= concated_FP
# np.save(f"count_exact_r0_to_r{num_plain_FPs-1}_FP.npy", count)

# %%
generate_FP_indices_of_r0_r15("test", 6144, "normal", dataset="2d")
generate_FP_indices_of_r0_r15("val", 6144, "normal", dataset="2d")
generate_FP_indices_of_r0_r15("test", 6144, "normal", dataset="1d")
generate_FP_indices_of_r0_r15("val", 6144, "normal", dataset="1d")


generate_FP_indices_of_r0_r15("test", 1024, "exact", dataset="2d")
generate_FP_indices_of_r0_r15("val", 1024, "exact", dataset="2d")
generate_FP_indices_of_r0_r15("test", 1024, "exact", dataset="1d")
generate_FP_indices_of_r0_r15("val", 1024, "exact", dataset="1d")

print("done")
# DONE

# %%
FP_on_bits_6144_1d_train_normal = generate_FP_indices_of_r0_r15("train", 6144, "normal", dataset="1d")
FP_on_bits_6144_2d_train_normal = generate_FP_indices_of_r0_r15("train", 6144, "normal", dataset="2d")
print("DONE") 






# %%
# Done
# FP_on_bits_1024_2d_train_exact = generate_FP_indices_of_r0_r15("train", 1024, "exact", dataset="2d")
# FP_on_bits_1024_1d_train_exact = generate_FP_indices_of_r0_r15("train", 1024, "exact", dataset="1d")


