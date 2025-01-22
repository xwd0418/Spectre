
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import pathlib, pickle, os, tqdm
import torch
import multiprocessing, collections

from collections import defaultdict  

RADIUS_UPPER_LIMIT = 4
DATASET_root_path = pathlib.Path("/workspace/")
DATASETS = ["OneD_Only_Dataset", "SMILES_dataset"]
DATASET_INDEX_SOURCE = ["oneD_NMR" , "HSQC"]

# step 1: find all fragments of the entire training set
def count_circular_substructures(smiles, radius=RADIUS_UPPER_LIMIT):
    # Example molecule
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    # Compute Morgan fingerprint with radius 
    info = {}
    fp = rdMolDescriptors.GetMorganFingerprint(mol, radius=radius, bitInfo=info)
    # Extract circular subgraphs
    circular_substructures = defaultdict(int) # radius to smiles
    # display(info)
    for bit_id, atom_envs in info.items():
        for atom_idx, curr_radius in atom_envs:
            # Get the circular environment as a subgraph
            env = Chem.FindAtomEnvironmentOfRadiusN(mol, curr_radius, atom_idx)
            submol = Chem.PathToSubmol(mol, env)
            smiles = Chem.MolToSmiles(submol, canonical=True)
            
            circular_substructures[smiles] += 1
                
    return circular_substructures
def count_words_in_file(f):
    idx = int(f.split(".")[0])
    smile = smiles_dict[idx]
    return count_circular_substructures(smile)

def merge_counts(counts_list):
    """Merges multiple word count dictionaries."""
    total_count = collections.Counter()
    for count in counts_list:
        total_count.update(count)
    return total_count

def get_all_train_set_fragments():
    # Load the dataset

    for dataset, index_souce in zip(DATASETS, DATASET_INDEX_SOURCE):
        global smiles_dict
        smiles_dict = pickle.load(open( DATASET_root_path / f"{dataset}/train/SMILES/index.pkl", "rb"))
        files = os.listdir( DATASET_root_path / f"{dataset}/train/{index_souce}")
        
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = []
        for count in tqdm.tqdm(pool.imap_unordered(count_words_in_file, files), total=len(files), desc="Processing files"):
            results.append(count)
        
    final_frags_count = merge_counts(results)
       
    # save a dictionary of fragment to index
    save_path = DATASET_root_path / f"count_fragments_radius_under_{RADIUS_UPPER_LIMIT}.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(final_frags_count, f)
   
if __name__ == "__main__":
    # print(get_sub_graphs("OCC1OC(OC2C(CO)OC(OC3C(CO)OC(OC4C(CO)OC(O)C(O)C4O)C(O)C3O)C(O)C2O)C(O)C(O)C1O"))
    get_all_train_set_fragments()
    