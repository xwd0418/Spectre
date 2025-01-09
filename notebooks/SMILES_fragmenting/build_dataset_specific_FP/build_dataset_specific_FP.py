
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import pathlib, pickle, os, tqdm

RADIUS_UPPER_LIMIT = 6
DATASET_root_path = pathlib.Path("/workspace/")
DATASETS = ["OneD_Only_Dataset", "SMILES_dataset"]
DATASET_INDEX_SOURCE = ["oneD_NMR" , "HSQC"]

# step 1: find all fragments of the entire training set
def get_sub_graphs(smiles):
    m = Chem.MolFromSmiles(smiles)
    # m = Chem.AddHs(m) # not needed because we look at SMILES, not like generate_FPs
    all_subgraphs_smiles = set()
    for radius in range(RADIUS_UPPER_LIMIT):
        subgraphs = Chem.FindUniqueSubgraphsOfLengthN(m, radius)
        for subgraph in subgraphs:
            # Get the subgraph as a new molecule
            submol = Chem.PathToSubmol(m, subgraph)
            smiles = Chem.MolToSmiles(submol, canonical=True)
            all_subgraphs_smiles.add(smiles)
            # print(smiles)
    return all_subgraphs_smiles

def get_all_train_set_fragments():
    # Load the dataset
    all_train_set_fragments = set()

    for dataset, index_souce in zip(DATASETS, DATASET_INDEX_SOURCE):
        smiles_dict = pickle.load(open( DATASET_root_path / f"{dataset}/train/SMILES/index.pkl", "rb"))
        file_names = os.listdir( DATASET_root_path / f"{dataset}/train/{index_souce}")
        for f in (pbar:=tqdm.tqdm(file_names)):
            idx = int(f.split(".")[0])
            smile = smiles_dict[idx]
            # print(smile)
            all_train_set_fragments.update(get_sub_graphs(smile))
            pbar.set_description(f"all_train_set_fragments {len(all_train_set_fragments)}")
    
    save_path = DATASET_root_path / f"all_train_set_fragments_up_to_radius_{RADIUS_UPPER_LIMIT}.pkl"
    pickle.dump(all_train_set_fragments, open(save_path, "wb"))
    




# step 2: generate FP for each molecule, based on which fragments are present in the molecule

if __name__ == "__main__":
    # print(get_sub_graphs("OCC1OC(OC2C(CO)OC(OC3C(CO)OC(OC4C(CO)OC(O)C(O)C4O)C(O)C3O)C(O)C2O)C(O)C(O)C1O"))
    get_all_train_set_fragments()
    