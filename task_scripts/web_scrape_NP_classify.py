import  requests, json,pickle,tqdm, os, sys
import pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed
spectre_dir = pathlib.Path(__file__).resolve().parents[1] 
sys.path.insert(0, str(spectre_dir))
print(spectre_dir)
from utils.get_NP_class import get_superclass_and_glycoside


def process_file(file, smiles_pkl):
    file_index = int(file.split('.')[0])
    smiles = smiles_pkl[file_index]
    superclass_results, isglycoside = get_superclass_and_glycoside(smiles)

    return file_index, superclass_results, isglycoside


if __name__ == "__main__":
    for split in [
        'val',
        # 'test',
        "train"
        ]:
        superclass_results = {}
        smiles_pkl = pickle.load(open(f'/workspace/SMILES_dataset/{split}/SMILES/index.pkl','rb'))
        # for k,v in tqdm.tqdm(smiles_pkl.items()):
    
        all_info_test_set_path = spectre_dir / f"datasets/{split}_indices_of_full_info_NMRs.pkl"
        all_info_test_set = pickle.load(open(all_info_test_set_path, 'rb'))

        
        with ThreadPoolExecutor(max_workers=8) as executor:  # Adjust max_workers based on network constraints
            future_to_file = {executor.submit(process_file, file, smiles_pkl): file for file in all_info_test_set}

            for future in tqdm.tqdm(as_completed(future_to_file), total=len(future_to_file)):
                file_index, superclass_results[file_index], isglycoside = future.result()
                
        save_path = f"/workspace/SMILES_dataset/{split}/Superclass/index.pkl"
        os.mkdir(f"/workspace/SMILES_dataset/{split}/Superclass")
        print(f"Saving to {save_path}")
        # print(superclass_results)
        with open(save_path, 'wb') as f:
            pickle.dump(superclass_results, f)