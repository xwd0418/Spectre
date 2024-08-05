# %%
"""
building dataset with the shortened FP and full FP from r0-r4, size 0f 30720 and reduce to 6144
"""

'''
Changed on 2024/4/19:
Using EXACT radius when generating the FP, from r0-r6

Note on 2024/4/22:
Can Configure which bit positions to pock
'''

import time, torch, os, pickle
from fingerprint_utils import FP_generator
import tqdm
from rdkit.Chem import AllChem
from rdkit import Chem
import numpy as np


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
   
  
def generate_FPs_and_concat(smile_str, num_FPs , exact_radius = False):
    generate_FP = generate_FP_with_exact_radius if exact_radius else generate_normal_FP
    
    mol = Chem.MolFromSmiles(smile_str)
    mol_H = Chem.AddHs(mol) # add implicit Hs to the molecule
    all_plain_fps = []
    for radius in range(num_FPs):
        all_plain_fps.append(generate_FP(mol_H, radius=radius, length=6144))
    concated_FP = np.concatenate(all_plain_fps)
    return torch.tensor(concated_FP)


if __name__ == "__main__":

    # MOST IMPORTANT PART: CONFIGURATIONS
    previously_computed_indices_to_keep = np.load("/root/MorganFP_prediction/reproduce_previous_works/Spectre/notebooks/dataset_building/indices_kept_r0_to_r4.npy")
    new_FP_name =  "R0_to_R4_reduced_FP"
    num_FPs = 5 # from r0 to r?
    dataset_to_add_FP = "OneD_Only_Dataset"
    sub_dir_of_all_molecules = "oneD_NMR" # "HYUN_FP"
    # END OF CONFIGURATIONS



    ### Build entropy based FP for SMILES dataset
    for split in ["test", "train", "val"]:
        # os.makedirs(f'/workspace/{dataset_to_add_FP}/{split}/R0_to_R6_exact_R_concat_FP/', exist_ok=True)
        os.makedirs(f'/workspace/{dataset_to_add_FP}/{split}/{new_FP_name}/', exist_ok=True)
        
        file_names = os.listdir( f"/workspace/{dataset_to_add_FP}/{split}/{sub_dir_of_all_molecules}/")
        path_dir = f"/workspace/{dataset_to_add_FP}/{split}/"
        smiles_dict = pickle.load(open(f"{path_dir}/SMILES/index.pkl", "rb"))

        for f in tqdm.tqdm(file_names):
            idx = int(f.split(".")[0])
            smile = smiles_dict[idx]
            r0_to_rEND_fp = generate_FPs_and_concat(smile,num_FPs,exact_radius = False)
            # torch.save(r0_to_rEND_fp, f'/workspace/{dataset_to_add_FP}/{split}/R0_to_R6_exact_R_concat_FP/{f}')
            
            torch.save(r0_to_rEND_fp[previously_computed_indices_to_keep], f'/workspace/{dataset_to_add_FP}/{split}/{new_FP_name}/{f}')
        
        
            



