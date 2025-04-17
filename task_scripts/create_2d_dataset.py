""" 
deprecated ... not even ever being used 


process smart dataset"""

import os, torch, pickle, json, random
from rdkit import Chem
from tqdm import tqdm
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List


# this pickle file is created by MorganFP_prediction/task_scripts/create_1D2D_dataset.ipynb
dataset_metadata = pickle.load(open("/root/gurusmart/data/Combine1D2D_dataset/dataset_metadata.pkl", 'rb')) 
data_split_dict = dataset_metadata['data_split_dict']

def smiles_to_inchi_key(smile):
    molecule = Chem.MolFromSmiles(smile)
    # Convert the RDKit molecule to an InChI key
    inchi_key = Chem.inchi.MolToInchiKey(molecule)
    return inchi_key

with open('/root/gurusmart/MorganFP_prediction/James_dataset_zips/SMART_dataset_v2_new.pkl', 'rb') as file:
    smart_v2_data = pickle.load(file)
hyun_smiles = [d['SMILES'] for d in smart_v2_data.values()]
hyun_HSQC = [d['HSQC'] for d in smart_v2_data.values()]
smart_inchi_keys = [smiles_to_inchi_key(s) for s in (hyun_smiles)]
empty_inchi_key_indices = [i for i, x in enumerate(smart_inchi_keys) if x == ""]

cleaned_smart_HSQC = remove_elements_by_index(hyun_HSQC, empty_inchi_key_indices)
cleaned_smart_inchikeys = remove_elements_by_index(smart_inchi_keys, empty_inchi_key_indices)


# Function to save a tensor - this is a blocking function
def save_tensor(tensor: torch.Tensor, file_name: str):
    torch.save(tensor, file_name)

# Asynchronous wrapper function to save tensors
async def async_save_tensor(tensor: torch.Tensor, file_name: str, executor,progress):
    loop = asyncio.get_event_loop()
    # Run the blocking function in a separate thread
    await loop.run_in_executor(executor, save_tensor, tensor, file_name)
    progress.update(1)

# Main asynchronous function to save multiple tensors
async def save_multiple_tensors(tensors: List[torch.Tensor], file_names: List[str]):
    with ThreadPoolExecutor() as executor:
        # Create a list of async tasks for saving tensors
        with tqdm(total=len(tensors)) as progress:
            tasks = [async_save_tensor(tensor, file_name, executor, progress) for tensor, file_name in zip(tensors, file_names)]

            # Wait for all tasks to complete
            await asyncio.gather(*tasks)

if __name__ == "__main__":
    dataset_dir = '/root/gurusmart/data/Combine1D2D_dataset'
    for split in ['train', 'val', 'test']:
        for spectrum in ['HSQC', "one_D", "hyun_FP", "R2_FP"]
        os.makedirs()
    tensors = [torch.tensor(HSQC, dtype=torch.float32) for HSQC in cleaned_smart_HSQC]
    file_names = [f"{dataset_dir}/{data_split_dict[inchikey]}/HSQC/{inchikey}.pt" for inchikey in cleaned_smart_inchikeys]
    asyncio.run(save_multiple_tensors(tensors, file_names))
    