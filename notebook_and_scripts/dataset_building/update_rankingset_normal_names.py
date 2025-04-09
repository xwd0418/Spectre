
import pickle


with open(f'/root/gurusmart/MorganFP_prediction/inference_data/coconut_loutus_hyun_training/inference_metadata_latest_RDkit.pkl', 'rb') as file:
    smiles_and_names = pickle.load(file)
print(len(smiles_and_names))



from rdkit import Chem
import requests

def get_chemical_name(smiles):
    # Convert SMILES to PubChem CID
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/cids/JSON"
    response = requests.get(url)
    
    if response.status_code != 200:
        return "Error: Unable to fetch CID"

    data = response.json()
    if "IdentifierList" not in data or "CID" not in data["IdentifierList"]:
        return "No CID found"

    cid = data["IdentifierList"]["CID"][0]

    # Get chemical name from CID
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/JSON"
    response = requests.get(url)
    
    if response.status_code != 200:
        return "Error: Unable to fetch compound data"

    data = response.json()
    for prop in data["PC_Compounds"][0]["props"]:
        if prop["urn"]["label"] == "IUPAC Name":
            return prop["value"]["sval"]

    return "No chemical name found"



import concurrent.futures
from rdkit import Chem
import requests
from tqdm import tqdm

# Function to process each tuple
def process_tuple(data_tuple):
    smiles, name, mw, src = data_tuple
    if name == "No name" or len(name) > 20:
        chemical_name = get_chemical_name(smiles)
        if chemical_name != "No chemical name found":
            return (smiles, chemical_name, mw, "pubchem")
    
    return data_tuple

# Multi-threaded function to process a list of tuples
def process_list_of_tuples(data_list):
    results = []
    
    # Using ThreadPoolExecutor to handle multiple threads
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # results = list(executor.map(process_tuple, data_list))
        results = list(tqdm(executor.map(process_tuple, data_list), total=len(data_list), desc="Processing SMILES"))
    
    return results


# Process the list of tuples
smiles_and_names = process_list_of_tuples(smiles_and_names)



with open(f'/root/gurusmart/MorganFP_prediction/inference_data/coconut_loutus_hyun_training/inference_metadata_normal_names.pkl', 'wb') as file:
    pickle.dump(smiles_and_names, file)



