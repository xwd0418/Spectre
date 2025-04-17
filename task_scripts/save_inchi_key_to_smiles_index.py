# %%
'''should look at the py file of the same name'''

import os, torch, pickle, json, random
from rdkit import Chem
from tqdm import tqdm
from collections import defaultdict


# %%
def smiles_to_inchi_key(smile):
    molecule = Chem.MolFromSmiles(smile)

    # Convert the RDKit molecule to an InChI key
    inchi_key = Chem.inchi.MolToInchiKey(molecule)
    return inchi_key

# %%
# construct dict of inchi_key -> "split"+"num.pt"
inchi_key_to_file_path = defaultdict(list)
for split in ["test", "train", "val"]:
    smiles = pickle.load(open(f'/root/gurusmart/MorganFP_prediction/James_dataset_zips/SMILES_dataset/{split}/SMILES/index.pkl', 'rb'))
    for num,smile in smiles.items():
        inchi_key = smiles_to_inchi_key(smile)
        inchi_key_to_file_path[inchi_key].append(f"{split}/detailed_oneD_NMR/{str(num)}.pt")

# %%
f = open('/root/gurusmart/data/NP-MRD-dataset/NP-MRD_metadata/npmrd_natural_products.json')
data = json.load(f)
NP_to_inchikey = dict([[d['accession'] , d['inchikey']]for d in data['np_mrd']['natural_product']])

# %%


# %%
to_save = {
    "NP_to_inchikey":NP_to_inchikey,
    "inchi_key_to_file_path": inchi_key_to_file_path
}
with open("/root/gurusmart/MorganFP_prediction/task_scripts/inchi_mapping.pkl", 'wb') as file:
    pickle.dump(to_save, file)
