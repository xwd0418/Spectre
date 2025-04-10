
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



def get_bitInfos_for_each_atom_idx(SMILES):
    """

    Args:
        SMILES (_type_): 

    Returns:
        atom_to_bit_infos (dict): 
         mapping atom index to a list of tuples; each tuple has (bit id, atom symbol, frag_smiles, radius)
    """
    
    mol = Chem.MolFromSmiles(SMILES)
    if mol is None:
        print(f"Failed to parse {SMILES}")
        # raise ValueError(f"Failed to parse {smiles}")
        return None, None
    # Chem.Kekulize(mol, clearAromaticFlags=True)

    # Compute Morgan fingerprint with radius 
    fp = gen.GetSparseFingerprint(mol, additionalOutput=ao)
    info = ao.GetBitInfoMap()          
    
    atom_to_bit_infos = defaultdict(list)
    all_bit_infos = set()
    for bit_id, atom_envs in info.items():
        # print(f'\n {bit_id=} ')
        for atom_idx, curr_radius in atom_envs:
            # Get the circular environment as a subgraph
            env = Chem.FindAtomEnvironmentOfRadiusN(mol, curr_radius, atom_idx)
            submol = Chem.PathToSubmol(mol, env)
            smiles = Chem.MolToSmiles(submol, canonical=True) # this is canonical in terms of fragment, so it is related to the bond/atom index mapping
            
            bit_info = (bit_id, mol.GetAtomWithIdx(atom_idx).GetSymbol(), smiles, curr_radius)
            atom_to_bit_infos[atom_idx].append(bit_info)
            all_bit_infos.add(bit_info)
            # print(f'{bit_info} ', end="")
            
    return atom_to_bit_infos, all_bit_infos
    

# step 1: find all fragments of the entire training set
def count_circular_substructures(smiles):
    bit_info_counter = defaultdict(int) 
    # substrucure_radius = {}
    
    atom_to_bit_infos, all_bit_infos = get_bitInfos_for_each_atom_idx(smiles)
    if atom_to_bit_infos is None:
        return bit_info_counter
    
    
    for bit_info in all_bit_infos:
    
            bit_info_counter[bit_info] = 1 # inside each molecule, each fragment is either on or off
            # print(f'{smiles} ', end="")
            
    return bit_info_counter


# def count_frags_in_file(f):
#     idx = int(f.split(".")[0])
#     smile = smiles_dict[idx]
#     return count_circular_substructures(smile)


def save_frags_for_file(f):
    number_part = f.split('/')[-1]
    idx = int(number_part.split(".")[0])
    smile = smiles_dict[idx]
    bit_info_counter = count_circular_substructures(smile)
   
  
    torch.save(list(bit_info_counter.keys()), f)
    
    return None

def merge_counts(counts_list):
    """Merges multiple word count dictionaries."""
    total_count = collections.Counter()
    for count in counts_list:
        total_count.update(count)
        
    return total_count

# def get_all_train_set_fragments():
#     """deprecated, because we want to see the entire retrieval dataset"""
#     # Load the dataset
#     count_results = []
#     radius_results = []
    
#     for dataset, index_souce in zip(DATASETS, DATASET_INDEX_SOURCE):
#         global smiles_dict
#         smiles_dict = pickle.load(open( DATASET_root_path / f"{dataset}/train/SMILES/index.pkl", "rb"))
#         files = os.listdir( DATASET_root_path / f"{dataset}/train/{index_souce}")
        
#         with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
#             for count, radius_mapping in tqdm.tqdm(pool.imap_unordered(count_frags_in_file, files), total=len(files), desc="Processing files"):
#                 count_results.append(count)
#                 radius_results.append(radius_mapping)
            
#     final_frags_count, fianl_radius_mapping = merge_counts(count_results, radius_results)
       
#     # save a dictionary of fragment to index
#     save_path = DATASET_root_path / f"count_fragments_radius_under_{RADIUS_UPPER_LIMIT}.pkl"
#     with open(save_path, "wb") as f:
#         pickle.dump(final_frags_count, f)
        
#     save_path = DATASET_root_path / f"radius_mapping_radius_under_{RADIUS_UPPER_LIMIT}.pkl"
#     with open(save_path, "wb") as f:
#         pickle.dump(fianl_radius_mapping, f)

def get_all_fragments_from_all_smiles_in_retrieval_dataset():
    # Load the dataset
    count_results = []
    
    with open(f'/root/gurusmart/MorganFP_prediction/inference_data/coconut_loutus_hyun_training/inference_metadata_latest_RDkit.pkl', 'rb') as file:
        smiles_and_names = pickle.load(file)
        all_SMILES = [x[0] for x in smiles_and_names]
        
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        for count in tqdm.tqdm(pool.imap_unordered(count_circular_substructures, all_SMILES), total=len(all_SMILES), desc="Processing files"):
            count_results.append(count)
            
    final_frags_count = merge_counts(count_results)
       
    # save a dictionary of fragment to index
    save_path = DATASET_root_path / f"count_hashes_under_radius_{RADIUS_UPPER_LIMIT}.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(final_frags_count, f)
   
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
    # get_all_fragments_from_all_smiles_in_retrieval_dataset()
    
    generate_frags() # generate fragments detail for each smiles in our dataset
    # (count_circular_substructures('COC(=O)c1c(O)cc(OC)c(CC=C(C)CCC=C(C)C)c1O'))