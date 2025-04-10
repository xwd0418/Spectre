
import numpy as np, os, torch, pickle, json, sys

with open(f'/root/gurusmart/MorganFP_prediction/inference_data/coconut_loutus_hyun_training/inference_metadata_latest_RDkit.pkl', 'rb') as file:
    smiles_and_names = pickle.load(file)
print(len(smiles_and_names))


hyunwoo_retrieve_data_path="/root/gurusmart/MorganFP_prediction/cleaned_dataset/DB_010621_SM3.json"
hyun_DB = json.load(open(hyunwoo_retrieve_data_path, 'r'))
print(len(hyun_DB))


all_smiles_in_my_data = set(x[0] for x in smiles_and_names)

all_HYUN_smiles = set(x["SMILES"] for x in hyun_DB)

def canonical(SMILES):
    from rdkit import Chem
    mol = Chem.MolFromSmiles(SMILES)
    smiles = Chem.MolToSmiles(mol, False) if mol is not None else None
    return smiles

all_hyun_canonical_smiles = set(x["SMILES"] for x in hyun_DB).intersection(all_smiles_in_my_data) .union( {canonical(x) for x in (all_HYUN_smiles - all_smiles_in_my_data)})


hyun_db_indices_in_my_data = []
for i, elements in enumerate(smiles_and_names):
    smiles = elements[0]
    if smiles in all_hyun_canonical_smiles:
        
        
        hyun_db_indices_in_my_data.append(i)
print(len(hyun_db_indices_in_my_data))


hyun_db_indices_in_my_data = torch.tensor(hyun_db_indices_in_my_data)


fp_root_dir = "/root/gurusmart/MorganFP_prediction/inference_data/inference_rankingset_with_stable_sort/"
for fp_type in os.listdir(fp_root_dir):
    
    load_path = os.path.join(fp_root_dir, fp_type, "FP_normalized.pt")
    FPs = torch.load(load_path).to_dense()
    torch.save(FPs[hyun_db_indices_in_my_data].to_sparse_csr(), os.path.join(fp_root_dir, fp_type, "FP_normalized_deepsat_retrieval_set.pt"))
    print("just saved", fp_type, FPs[hyun_db_indices_in_my_data].shape)




