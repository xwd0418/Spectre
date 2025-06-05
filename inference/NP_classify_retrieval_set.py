# %%
import requests
def fetch_np_class(smile):
    try:
        np_url = f"https://npclassifier.gnps2.org/classify?smiles={smile}"
        res = requests.get(np_url, timeout=5)
        return res.json()
    except Exception as e:
        print(f"[WARNING] NPClassifier failed for {smile}: {e}")
        return {"error": "NPClassifier request failed"}
    
fetch_np_class("CCC(C)C(OC(=O)C(CC(C)C)N(C)C)C(=O)N(C)C(C(=O)N1CCCC1C(=O)N1C(=O)C=C(OC)C1C(C)C)C(C)C")

# %%
import pickle
with open(f'/root/gurusmart/MorganFP_prediction/inference_data/coconut_loutus_hyun_training/inference_metadata_latest_RDkit.pkl', 'rb') as file:
    smiles_and_names = pickle.load(file)
print(len(smiles_and_names))

# %%
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Define your fetch function and input list
def fetch_with_key(smiles_entry):
    smiles, _, _, _ = smiles_entry
    result = fetch_np_class(smiles)
    return smiles, result

# Run multiprocessing
def run_parallel(smiles_and_names, num_workers=8):
    smiles_to_np_classes = {}
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(fetch_with_key, entry): entry for entry in smiles_and_names}
        for future in tqdm(as_completed(futures), total=len(futures)):
            smiles, result = future.result()
            smiles_to_np_classes[smiles] = result
    return smiles_to_np_classes




# %%
# Run and save
smiles_to_np_classes = run_parallel(smiles_and_names, num_workers=16)

with open('/root/gurusmart/MorganFP_prediction/inference_data/coconut_loutus_hyun_training/smiles_to_np_classes.pkl', 'wb') as file:
    pickle.dump(smiles_to_np_classes, file)


