# %%
"""build inference dataset for the model

using the isomeric SMILES without stereochemistry information
"""
import torch, pickle
import sys, os 


''' build entropy-based-fp '''
import os 
import torch.nn.functional as F

from multiprocessing import Pool, cpu_count

import tqdm
import sys

sys.path.insert(0,"/root/gurusmart/MorganFP_prediction")


fp_dim = 16384
max_radius = 6

fp_loader = None

from datasets.fp_loader_utils import  FP_Loader_Configer
def init_worker():
    """Initialize a separate fp_loader for each worker process."""
    global fp_loader
    
    # Set up FP loader
    fp_loader_configer = FP_Loader_Configer()
    fp_loader_configer.select_version("Hash_Entropy")
    fp_loader = fp_loader_configer.fp_loader
    fp_loader.setup(out_dim=fp_dim, max_radius=max_radius)
    
    
    print(f"Initialized fp_loader in process ")

def process_smiles_batch(batch):
    """Each process uses its own fp_loader to compute fingerprints."""
    global fp_loader
    built_fps = [fp_loader.build_mfp_for_new_SMILES(smile_and_other_info[0]) for smile_and_other_info in batch]
    return F.normalize(torch.stack(built_fps).float(), dim=1, p=2.0)


def step1_build_small_pieces():
    save_path = '/root/gurusmart/MorganFP_prediction/inference_data/coconut_loutus_hyun_training/inference_metadata_latest_RDkit.pkl'
    all_smiles_and_chemical_names = pickle.load(open(save_path, 'rb'))

    save_dir = f"/root/gurusmart/MorganFP_prediction/inference_data/inference_rankingset_with_stable_sort/non_collision_FP_rankingset_max_radius_{max_radius}_dim_{fp_dim}/"
    os.makedirs(save_dir, exist_ok=True)

    batch_size = 500
    num_workers = min(cpu_count(), 4)  # Use up to 8 cores

    # Split data into batches
    batches = [all_smiles_and_chemical_names[i:i + batch_size] for i in range(0, len(all_smiles_and_chemical_names), batch_size)]

 

    with Pool(num_workers, initializer=init_worker) as pool:
        for file_number, result in enumerate(tqdm.tqdm(pool.imap(process_smiles_batch, batches), total=len(batches))):
            save_path = os.path.join(save_dir, f'FP_{file_number}.pt')
            torch.save(result, save_path)

    print(f"Step1 finished: Built entropy-based FP with max_radius={max_radius} and fp_dim={fp_dim}")

# %%
'''stack little pieces (2000) toghether'''
def step2_stack_small_pieces():
    print('start stacking little pieces together')
    
    # save_dir = f"/root/gurusmart/MorganFP_prediction/inference_data/inference_rankingset_with_stable_sort/db_specific_FP_rankingset_max_radius_{max_radius}_fp_dim_{fp_dim}/"
    save_dir = f"/root/gurusmart/MorganFP_prediction/inference_data/inference_rankingset_with_stable_sort/non_collision_FP_rankingset_max_radius_{max_radius}_dim_{fp_dim}/"
    all_file = os.listdir(save_dir)
    all_file.sort(key=lambda x:int(x.split("_")[1].split(".")[0]))
    
    everything = []
    for f in tqdm.tqdm(all_file):
        x = torch.load(save_dir + f)
        everything.append(x)
        
    out = torch.vstack(everything)
    print(out.shape)
    # save_dir_together = f"/root/gurusmart/MorganFP_prediction/inference_data/inference_rankingset_with_stable_sort/db_specific_FP_rankingset_max_radius_{max_radius}_fp_dim_{fp_dim}_stacked_together/"
    save_dir_together = f"/root/gurusmart/MorganFP_prediction/inference_data/inference_rankingset_with_stable_sort/non_collision_FP_rankingset_max_radius_{max_radius}_dim_{fp_dim}_stacked_together/"
    os.makedirs(save_dir_together, exist_ok=True)
    torch.save(out.to_sparse_csr(),  save_dir_together+ 'FP.pt')
    print("step2 finished")
    print(f'finished stacking max_radius_{max_radius}')


if __name__ == '__main__':
    if os.environ.get('step') == '1':
        step1_build_small_pieces()
    if os.environ.get('step') == '2':
        step2_stack_small_pieces()