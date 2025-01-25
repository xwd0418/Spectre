
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import pathlib, pickle, os, tqdm, torch
import multiprocessing, collections

from collections import defaultdict  

RADIUS_UPPER_LIMIT = 10
DATASET_root_path = pathlib.Path("/workspace/")
DATASETS = ["OneD_Only_Dataset", "SMILES_dataset"]
DATASET_INDEX_SOURCE = ["oneD_NMR" , "HSQC"]

from rdkit.Chem import rdFingerprintGenerator
gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS_UPPER_LIMIT)
ao = rdFingerprintGenerator.AdditionalOutput()
ao.AllocateBitInfoMap()

# step 1: find all fragments of the entire training set
def count_circular_substructures(smiles):
    # Example molecule
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Failed to parse {smiles}")
        raise ValueError(f"Failed to parse {smiles}")
    mol = Chem.AddHs(mol)

    # Compute Morgan fingerprint with radius 
    fp = gen.GetFingerprint(mol, additionalOutput=ao)
    info = ao.GetBitInfoMap()

    # Extract circular subgraphs
    circular_substructures_counts = defaultdict(int) # radius to smiles
    substrucure_radius = {}
    # display(info)
    for bit_id, atom_envs in info.items():
        for atom_idx, curr_radius in atom_envs:
            # Get the circular environment as a subgraph
            env = Chem.FindAtomEnvironmentOfRadiusN(mol, curr_radius, atom_idx)
            submol = Chem.PathToSubmol(mol, env)
            smiles = Chem.MolToSmiles(submol, canonical=True)
            
            circular_substructures_counts[smiles] = 1
            if smiles not in substrucure_radius:
                substrucure_radius[smiles] = curr_radius
            else:
                substrucure_radius[smiles] = min(substrucure_radius[smiles], curr_radius)
    return circular_substructures_counts, substrucure_radius


def count_frags_in_file(f):
    idx = int(f.split(".")[0])
    smile = smiles_dict[idx]
    return count_circular_substructures(smile)


def save_frags_for_file(f):
    number_part = f.split('/')[-1]
    idx = int(number_part.split(".")[0])
    smile = smiles_dict[idx]
    try:
        count, _ = count_circular_substructures(smile)
    except ValueError:
        print(f"Failed to parse {f}\n\n")
        exit(0)
        return None
  
    torch.save(list(count.keys()), f)
    
    return None

def merge_counts(counts_list, radius_list):
    """Merges multiple word count dictionaries."""
    total_count = collections.Counter()
    for count in counts_list:
        total_count.update(count)
        
    min_radius = {}
    for radius_map in radius_list:
        for k, v in radius_map.items():
            if k not in min_radius:
                min_radius[k] = v
            else:
                min_radius[k] = min(min_radius[k], v)
    return total_count, min_radius

def get_all_train_set_fragments():
    # Load the dataset

    count_results = []
    radius_results = []
    
    for dataset, index_souce in zip(DATASETS, DATASET_INDEX_SOURCE):
        global smiles_dict
        smiles_dict = pickle.load(open( DATASET_root_path / f"{dataset}/train/SMILES/index.pkl", "rb"))
        files = os.listdir( DATASET_root_path / f"{dataset}/train/{index_souce}")
        
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            for count, radius_mapping in tqdm.tqdm(pool.imap_unordered(count_frags_in_file, files), total=len(files), desc="Processing files"):
                count_results.append(count)
                radius_results.append(radius_mapping)
            
    final_frags_count, fianl_radius_mapping = merge_counts(count_results, radius_results)
       
    # save a dictionary of fragment to index
    save_path = DATASET_root_path / f"count_fragments_radius_under_{RADIUS_UPPER_LIMIT}.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(final_frags_count, f)
        
    save_path = DATASET_root_path / f"radius_mapping_radius_under_{RADIUS_UPPER_LIMIT}.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(fianl_radius_mapping, f)
        
        
def generate_frags():

    for dataset, index_souce in zip(DATASETS, DATASET_INDEX_SOURCE):
        for split in ["test", "val", "train"]:
            os.makedirs(DATASET_root_path / f"{dataset}/{split}/fragments_of_different_radii", exist_ok=True)
            global smiles_dict
            smiles_dict = pickle.load(open( DATASET_root_path / f"{dataset}/{split}/SMILES/index.pkl", "rb"))
            files = os.listdir( DATASET_root_path / f"{dataset}/{split}/{index_souce}")
            files = [ f"{DATASET_root_path}/{dataset}/{split}/fragments_of_different_radii/{f}" for f in files]
            
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                for _ in tqdm.tqdm(pool.imap_unordered(save_frags_for_file, files), total=len(files), desc=f"Processing files {dataset}/{split}/"):
                    pass
            
            
                    
   
if __name__ == "__main__":
    get_all_train_set_fragments()
    generate_frags()