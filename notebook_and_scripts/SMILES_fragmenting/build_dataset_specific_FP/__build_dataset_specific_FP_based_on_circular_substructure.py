
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import pathlib, pickle, os, tqdm
import torch
from collections import defaultdict  

RADIUS_UPPER_LIMIT = 4
DATASET_root_path = pathlib.Path("/workspace/")
DATASETS = ["OneD_Only_Dataset", "SMILES_dataset"]
DATASET_INDEX_SOURCE = ["oneD_NMR" , "HSQC"]

# step 1: find all fragments of the entire training set
def get_circular_substructures(smiles, radius=RADIUS_UPPER_LIMIT):
    # Example molecule
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    # Compute Morgan fingerprint with radius 
    info = {}
    fp = rdMolDescriptors.GetMorganFingerprint(mol, radius=radius, bitInfo=info)
    # Extract circular subgraphs
    circular_substructures = defaultdict(set) # radius to smiles
    # display(info)
    for bit_id, atom_envs in info.items():
        for r in range(radius+1):
            for atom_idx, curr_radius in atom_envs:
                if curr_radius != r:
                    continue
                # Get the circular environment as a subgraph
                env = Chem.FindAtomEnvironmentOfRadiusN(mol, curr_radius, atom_idx)
                submol = Chem.PathToSubmol(mol, env)
                smiles = Chem.MolToSmiles(submol, canonical=True)
                # if smiles in circular_substructures:
                #     print(f"Already found: {smiles}")
                
                circular_substructures[curr_radius].add(smiles)
                
    return circular_substructures

def get_all_train_set_fragments():
    # Load the dataset
    all_train_set_fragments = defaultdict(set)
    all_frags = set()

    for dataset, index_souce in zip(DATASETS, DATASET_INDEX_SOURCE):
        smiles_dict = pickle.load(open( DATASET_root_path / f"{dataset}/train/SMILES/index.pkl", "rb"))
        file_names = os.listdir( DATASET_root_path / f"{dataset}/train/{index_souce}")
        for f in (pbar:=tqdm.tqdm(file_names)):
            idx = int(f.split(".")[0])
            smile = smiles_dict[idx]
            # print(smile)
            circular_substructures = get_circular_substructures(smile)
            for r, smiles_set in circular_substructures.items():
                new_smiles = smiles_set - all_frags
                all_train_set_fragments[r].update(new_smiles)
                all_frags.update(new_smiles)
            pbar.set_description(f"all_train_set_fragments {sum([len(v) for v in all_train_set_fragments.values()])}")
    
        
    # save a dictionary of fragment to index
    save_path = DATASET_root_path / f"all_train_set_fragments_radius_under_{RADIUS_UPPER_LIMIT}_to_smiles.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(all_train_set_fragments, f)
    
    # save dicts converting smiles and index back and forth
    radius_to_smiles_to_index = defaultdict(dict)
    index_to_smiles = {}
    counter = 0
    for r in sorted(all_train_set_fragments):
        for frag_smiles in all_train_set_fragments[r]:
            radius_to_smiles_to_index[r][frag_smiles] = counter
            index_to_smiles[counter] = frag_smiles
            counter += 1 
               
    gen_FP_save_path = DATASET_root_path / f"all_train_set_fragments_radius_to_smiles_to_index.pkl" # used for generating FP
    with open(gen_FP_save_path, "wb") as f: 
        pickle.dump(radius_to_smiles_to_index, f)
        
    interpret_FP_save_path = DATASET_root_path / f"all_train_set_fragments_index_to_smiles.pkl" # used for interpreting FP
    with open(interpret_FP_save_path, "wb") as f: 
        pickle.dump(index_to_smiles, f)
        




# step 2: generate FP for each molecule, based on which fragments are present in the molecule
def generate_FPs():
    all_train_set_fragments = pickle.load(open(DATASET_root_path / f"all_train_set_fragments_radius_under_{RADIUS_UPPER_LIMIT}_to_smiles.pkl", "rb"))
    radius_to_smiles_to_index = pickle.load(open(DATASET_root_path / f"all_train_set_fragments_radius_to_smiles_to_index.pkl", "rb"))
    
    for dataset, index_souce in zip(DATASETS, DATASET_INDEX_SOURCE):
        for split in ["val", "train", "test"]:
            os.makedirs(DATASET_root_path / f"{dataset}/{split}/DB_specific_FP", exist_ok=True)
            smiles_dict = pickle.load(open( DATASET_root_path / f"{dataset}/{split}/SMILES/index.pkl", "rb"))
            file_names = os.listdir( DATASET_root_path / f"{dataset}/{split}/{index_souce}")
            for f in (pbar:=tqdm.tqdm(file_names)):
                idx = int(f.split(".")[0])
                smile = smiles_dict[idx]
                sub_smiles = get_circular_substructures(smile)
                
                output_FP_dict = {}
                for r, smiles_set in sub_smiles.items():
                    bits_on = [radius_to_smiles_to_index[r][s] for s in (smiles_set.intersection(all_train_set_fragments[r]))]
                    output_FP_dict[r] = torch.tensor(bits_on)
                
                torch.save(output_FP_dict, DATASET_root_path / f"{dataset}/{split}/DB_specific_FP/{idx}.pt")
                    

if __name__ == "__main__":
    # print(get_sub_graphs("OCC1OC(OC2C(CO)OC(OC3C(CO)OC(OC4C(CO)OC(O)C(O)C4O)C(O)C3O)C(O)C2O)C(O)C(O)C1O"))
    get_all_train_set_fragments()
    generate_FPs()
    