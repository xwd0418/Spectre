import numpy as np, pickle
import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

import sys, pathlib

repo_path = pathlib.Path(__file__).resolve().parents[1]
DATASET_root_path = pathlib.Path("/workspace/")


# utils about entropy and MFP
def compute_entropy(data, total_dataset_size, use_natural_log=False):
    # data: an array of counts of each bit, by default in the size of self.out_dim*(max_radius+1)
    # print(type(data), data.dtype)
    # print(type(total_dataset_size))
    probability = data/total_dataset_size
    p = probability
    # if use_natural_log:
    #     entropy = (probability * np.log(np.clip(probability,1e-7 ,1)) )
    # else:
    #     entropy = (probability * np.log2(np.clip(probability,1e-7 ,1)) )
    if use_natural_log:
        entropy = p * np.log(np.clip(p,1e-7 ,1))  +  (1-p) * np.log(np.clip(1-p,1e-7 ,1))
    else:
        entropy = p * np.log2(np.clip(p,1e-7 ,1))  +  (1-p) * np.log2(np.clip(1-p,1e-7 ,1))
    print("finish entropy list")
    return entropy

def keep_smallest_entropy(data, total_dataset_size, size,  use_natural_log=False):
    entropy = compute_entropy(data, total_dataset_size, use_natural_log)
    indices_of_min_6144 = np.argsort(entropy, kind="stable")[:size]
    # print(entropy, indices_of_min_6144)
    total_entropy = entropy[indices_of_min_6144]
    return total_entropy, indices_of_min_6144

''' 
convert the on bit positions to an array of 0s and 1s
'''
def convert_bits_positions_to_array(FP_on_bits, length):
    FP_on_bits= FP_on_bits.astype(int)
    out = np.zeros(length)
    out[FP_on_bits] = 1
    return out


def generate_morgan_FP(mol, radius=2, length=6144):
    if type(mol) == str:
        mol = Chem.MolFromSmiles(mol)
    mol = Chem.AddHs(mol) # add implicit Hs to the molecule
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=length)
 
    fp = gen.GetFingerprint(mol)
    return np.array(fp)



def generate_morgan_FP_on_bits(mol, radius=2, length=6144):
    if type(mol) == str:
        mol = Chem.MolFromSmiles(mol)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=length)
 
    fp = gen.GetFingerprint(mol)
    return np.array(fp.GetOnBits())

# define abstract class for FP loader
class FP_loader():
    def __init__(self, ) -> None:
        pass
    def setup(self, ):
        pass
    def build_mfp(self, ):
        pass
    
    def build_rankingset(self, split, predefined_FP = None):         
        # assuming rankingset on allinfo-set
        path_to_load_full_info_indices = f"{repo_path}/datasets/{split}_indices_of_full_info_NMRs.pkl"
        file_idx_for_ranking_set = pickle.load(open(path_to_load_full_info_indices, "rb"))

        files  = [self.build_mfp(int(file_idx.split(".")[0]), "2d", split) for file_idx in sorted(file_idx_for_ranking_set)]
        out = torch.vstack(files)
        return out
    
    def build_mfp_for_new_SMILES(self, ):
        pass
    def build_inference_ranking_set_with_everything(self, fp_dim, max_radius, use_hyun_fp = False, test_on_deepsat_retrieval_set = False):
        pass
    
# @set_float32_highest_precision
class Specific_Radius_MFP_loader(FP_loader):
    def __init__(self, ) -> None:
        self.only_2d = None
        self.out_dim = None
        self.max_radius = None
        # self.path_pickles = f'{repo_path}/notebook_and_scripts/dataset_building/FP_on_bits_pickles/'
        self.path_pickles = "/workspace/FP_on_bits_pickles/"
    def setup(self, only_2d = False, FP_building_type = "Normal", out_dim=6144):
        from  time import time
        time1 = time()
        assert (FP_building_type in ["Normal", "Exact"]), "Only Normal/Exact FP is supported"
        self.out_dim = out_dim
        self.only_2d = only_2d
        self.single_FP_size = 6144 if FP_building_type == "Normal" else 1024
        self.train_1d = pickle.load(open(self.path_pickles + f'{FP_building_type}_FP_on_bits_r0_r15_len_{self.single_FP_size}_1d_train.pkl', 'rb'))
        self.train_2d = pickle.load(open(self.path_pickles + f'{FP_building_type}_FP_on_bits_r0_r15_len_{self.single_FP_size}_2d_train.pkl', 'rb'))
        time2 = time()
        print(f"loading time: {time2-time1}")
        self.count = np.zeros(self.single_FP_size*16)
        if not only_2d:
            for FP_on_bits in self.train_1d.values():
                self.count += convert_bits_positions_to_array(FP_on_bits, self.single_FP_size*16)
            # self.count += np.sum([convert_bits_positions_to_array(FP_on_bits, self.single_FP_size*16) for FP_on_bits in self.train_1d.values()], axis=0)
        for FP_on_bits in self.train_2d.values():
            self.count += convert_bits_positions_to_array(FP_on_bits, self.single_FP_size*16)
        # self.count += np.sum([convert_bits_positions_to_array(FP_on_bits, self.single_FP_size*16) for FP_on_bits in self.train_2d.values()], axis=0)
        time3 = time()
        print(f"counting time: {time3-time2}")
        # also val and test
        self.val_1d = pickle.load(open(self.path_pickles + f'{FP_building_type}_FP_on_bits_r0_r15_len_{self.single_FP_size}_1d_val.pkl', 'rb'))
        self.test_1d = pickle.load(open(self.path_pickles + f'{FP_building_type}_FP_on_bits_r0_r15_len_{self.single_FP_size}_1d_test.pkl', 'rb'))
        self.val_2d = pickle.load(open(self.path_pickles + f'{FP_building_type}_FP_on_bits_r0_r15_len_{self.single_FP_size}_2d_val.pkl', 'rb'))
        self.test_2d = pickle.load(open(self.path_pickles + f'{FP_building_type}_FP_on_bits_r0_r15_len_{self.single_FP_size}_2d_test.pkl', 'rb'))
        
    
    def set_max_radius(self, max_radius, only_2d = False):
        self.max_radius = max_radius
        count_for_current_radius = self.count[:self.single_FP_size*(max_radius+1)]
        if only_2d:
            total_dataset_size = len(self.train_2d)
        else:
            total_dataset_size = len(self.train_1d) + len(self.train_2d)
        self.total_entropy_of_all_bits, self.indices_kept = keep_smallest_entropy(count_for_current_radius, total_dataset_size, self.out_dim)
        # self.indices_kept = list(self.indices_kept)
        assert len(self.indices_kept) == self.out_dim, f"should keep only {self.out_dim} highest entropy bits"
        
        
    def build_mfp(self, file_idx, dataset_src, split):
        if dataset_src == '1d':
            if split == 'train':
                FP_on_bits = self.train_1d[file_idx]
            elif split == 'val':
                FP_on_bits = self.val_1d[file_idx]
            elif split == 'test':
                FP_on_bits = self.test_1d[file_idx]
        elif dataset_src == '2d':
            if split == 'train':
                FP_on_bits = self.train_2d[file_idx]
            elif split == 'val':
                FP_on_bits = self.val_2d[file_idx]
            elif split == 'test':
                FP_on_bits = self.test_2d[file_idx]
        else:
            raise ValueError("dataset should be either 1d or 2d")
        
        mfp = convert_bits_positions_to_array(FP_on_bits, self.single_FP_size*16)
        mfp = mfp[self.indices_kept]
        return torch.tensor(mfp).float()
    
    def build_rankingset(self, split, predefined_FP = None):         
        path_to_load_full_info_indices = f"{repo_path}/datasets/{split}_indices_of_full_info_NMRs.pkl"
        file_idx_for_ranking_set = pickle.load(open(path_to_load_full_info_indices, "rb"))
        if predefined_FP == "HYUN_FP":
            dataset_path = DATASET_root_path / f"SMILES_dataset/{split}/{predefined_FP}"
            files  = [torch.load(F"{dataset_path}/{file_idx}").float() for file_idx in sorted(file_idx_for_ranking_set)]
        # elif type(predefined_FP) is str and predefined_FP.startswith("DB_specific_FP"):
        #     radius = int(predefined_FP[-1])
        #     files  = [self.build_db_specific_fp(int(file_idx.split(".")[0]), "2d", split, radius) for file_idx in sorted(file_idx_for_ranking_set)]
            
        else: # entropy-based FP
            files  = [self.build_mfp(int(file_idx.split(".")[0]), "2d", split) for file_idx in sorted(file_idx_for_ranking_set)]
             
        out = torch.vstack(files)
        return out
    
    def build_inference_ranking_set_with_everything(self, use_hyun_fp = False, test_on_deepsat_retrieval_set = False, fp_dim = None, max_radius = None):
        if use_hyun_fp:
            rankingset_path = "/root/gurusmart/MorganFP_prediction/inference_data/inference_rankingset_with_stable_sort/hyun_fp_stacked_together_sparse/FP_normalized.pt"
        else:
            rankingset_path = f"/root/gurusmart/MorganFP_prediction/inference_data/inference_rankingset_with_stable_sort/max_radius_{self.max_radius}_stacked_together_sparse/FP_normalized.pt"
        if test_on_deepsat_retrieval_set:
          
            rankingset_path = rankingset_path.replace("FP_normalized", "FP_normalized_deepsat_retrieval_set")
        print(f"loading {rankingset_path}")
        return torch.load(rankingset_path)#.to("cuda")
        
        
    def build_mfp_for_new_SMILES(self, smiles):
        try:
            num_plain_FPs = 16 # radius from 0 to 15
            
            mol = Chem.MolFromSmiles(smiles)
            mol_H = Chem.AddHs(mol) # add implicit Hs to the molecule
            all_plain_fps_on_bits = []
            for radius in range(num_plain_FPs):
                all_plain_fps_on_bits.append(generate_morgan_FP_on_bits(mol_H, radius=radius, length=self.single_FP_size) + radius*self.single_FP_size)
            FP_on_bits = np.concatenate(all_plain_fps_on_bits)
            
            mfp = convert_bits_positions_to_array(FP_on_bits, self.single_FP_size*16)
            mfp = mfp[self.indices_kept]
            return torch.tensor(mfp).float()
        except:
            return torch.zeros(self.out_dim).float()
    
# @set_float32_highest_precision
class DB_Specific_FP_loader(FP_loader):
    def __init__(self, ) -> None:
        
        RADIUS_UPPER_LIMIT = 10
        save_path = DATASET_root_path / f"count_fragments_radius_under_{RADIUS_UPPER_LIMIT}.pkl"
        with open(save_path, "rb") as f:
            self.frags_count = pickle.load(f)
        save_path = DATASET_root_path / f"radius_mapping_radius_under_{RADIUS_UPPER_LIMIT}.pkl"
        with open(save_path, "rb") as f:
            self.radius_mapping = pickle.load(f)
        self.max_radius = None
        self.out_dim = None
        
    def setup(self, out_dim, max_radius):
        if self.out_dim == out_dim and self.max_radius == max_radius:
            print("DB_Specific_FP_loader is already setup")
            return
        self.max_radius = max_radius
        frags_to_use_counts_and_smiles = [ [v,k] for k, v in self.frags_count.items() if self.radius_mapping[k] <= max_radius]
        counts, frag_smiles = zip(*frags_to_use_counts_and_smiles)
        counts = np.array(counts)
        frag_smiles = np.array(frag_smiles)
        if out_dim == 'inf' or out_dim == float("inf"):
            out_dim = len(frags_to_use_counts_and_smiles)
        self.out_dim = out_dim
         
        # create a N*2 array of  [entropy, smiles]
        entropy_each_frag = compute_entropy(counts, total_dataset_size = len(self.frags_count))
                                              
        frags_to_keep = frag_smiles[np.lexsort((frag_smiles, entropy_each_frag))[:out_dim]]   
        self.frag_to_index_map = {smiles: i for i, smiles in enumerate(frags_to_keep)}
        print(f"DB_Specific_FP_loader is setup, {out_dim=}, {max_radius=}")
        return entropy_each_frag, counts, len(self.frags_count)
        
    def build_mfp(self, file_idx, dataset_src, split):
        if dataset_src == "2d":
            dataset_dir = "SMILES_dataset"
        elif dataset_src == "1d":
            dataset_dir = "OneD_Only_Dataset"
        else:
            raise ValueError("dataset should be either 1d or 2d")
        dataset_path = DATASET_root_path / f"{dataset_dir}/{split}/fragments_of_different_radii"
        file_path = dataset_path / f"{file_idx}.pt"
        fragments = torch.load(file_path) 
        mfp = np.zeros(self.out_dim)
        for frag in fragments:
            if frag in self.frag_to_index_map:
                mfp[self.frag_to_index_map[frag]] = 1
        return torch.tensor(mfp).float()
    
    
    def build_rankingset(self, split, predefined_FP=None):
        return super().build_rankingset(split, predefined_FP)

    def build_inference_ranking_set_with_everything(self, fp_dim, max_radius, use_hyun_fp = False, test_on_deepsat_retrieval_set = False):
        if use_hyun_fp:
            rankingset_path = "/root/gurusmart/MorganFP_prediction/inference_data/inference_rankingset_with_stable_sort/hyun_fp_stacked_together_sparse/FP_normalized.pt" # FP and FP_normalized are the same
        else:
            rankingset_path = f"/root/gurusmart/MorganFP_prediction/inference_data/inference_rankingset_with_stable_sort/db_specific_FP_rankingset_max_radius_{max_radius}_fp_dim_{fp_dim}_stacked_together/FP.pt"
        if test_on_deepsat_retrieval_set:
            rankingset_path = rankingset_path.replace("FP_normalized", "FP_normalized_deepsat_retrieval_set")
        print(f"loading {rankingset_path}")
        return torch.load(rankingset_path)#.to("cuda")
    
    
    def build_mfp_for_new_SMILES(self, smiles):
        from notebook_and_scripts.SMILES_fragmenting.build_dataset_specific_FP.find_frags import count_circular_substructures
        mfp = np.zeros(self.out_dim)
        
        try:
            frags_with_count, _ = count_circular_substructures(smiles)
        except:
            return torch.tensor(mfp).float() # return all zeros if the SMILES is invalid, so cosine similarity will be 0
        for frag in frags_with_count:
            if frag in self.frag_to_index_map:
                mfp[self.frag_to_index_map[frag]] = 1
        
        return torch.tensor(mfp).float()
        
    def construct_index_to_frag_mapping(self):
        self.index_to_frag_mapping =  {v:k for k, v in self.frag_to_index_map.items()}


class Hash_Entropy_FP_loader(FP_loader):
    def __init__(self, ) -> None:
        RADIUS_UPPER_LIMIT = 10
        save_path = DATASET_root_path / f"count_hashes_under_radius_10.pkl"
        with open(save_path, "rb") as f:
            self.hashed_bits_count = pickle.load(f)
        self.max_radius = None
        self.out_dim = None
        
        
    def setup(self, out_dim, max_radius):
        if self.out_dim == out_dim and self.max_radius == max_radius:
            print("Hash_Entropy_FP_loader is already setup")
            return
        self.max_radius = max_radius
        filtered_bitinfos_and_their_counts = [((bit_id, atom_symbol, frag_smiles, radius), counts)  for (bit_id, atom_symbol, frag_smiles, radius), counts in self.hashed_bits_count.items() if radius <= max_radius]
        bitinfos, counts = zip(*filtered_bitinfos_and_their_counts)
        counts = np.array(counts)
        if out_dim == 'inf' or out_dim == float("inf"):
            out_dim = len(filtered_bitinfos_and_their_counts)
        self.out_dim = out_dim
         
        # create a N*2 array of  [entropy, smiles]
        # with open(f'/root/gurusmart/MorganFP_prediction/inference_data/coconut_loutus_hyun_training/inference_metadata_latest_RDkit.pkl', 'rb') as file:
        #     retrieval_set = pickle.load(file)
        # retrieval_set_size = len(retrieval_set)
        retrieval_set_size = 526316
        entropy_each_frag = compute_entropy(counts, total_dataset_size = retrieval_set_size)
                                        
        # print   (bitinfos.shape, counts.shape, entropy_each_frag.shape)
        # bit_infos_to_keep = bitinfos[np.argsort(entropy_each_frag, kind="stable")[:out_dim]]  
        # self.frag_to_index_map = {bit_info: i for i, bit_info in enumerate(bit_infos_to_keep)}
        indices_of_high_entropy = np.argsort(entropy_each_frag, kind="stable")[:out_dim]
        self.bitInfos_to_fp_index_map = {bitinfos[bitinfo_list_index]: fp_index for fp_index, bitinfo_list_index in enumerate(indices_of_high_entropy)}
        self.fp_index_to_bitInfo_mapping =  {v:k for k, v in self.bitInfos_to_fp_index_map.items()}
        print(f"Hash_Entropy_FP_loader is setup, {out_dim=}, {max_radius=}")
        # return entropy_each_frag, counts, len(self.frags_count)

    
    def build_mfp(self, file_idx, dataset_src, split):
        if dataset_src == "2d":
            dataset_dir = "SMILES_dataset"
        elif dataset_src == "1d":
            dataset_dir = "OneD_Only_Dataset"
        else:
            raise ValueError("dataset should be either 1d or 2d")
        dataset_path = DATASET_root_path / f"{dataset_dir}/{split}/fragments_of_different_radii"
        file_path = dataset_path / f"{file_idx}.pt"
        fragment_infos = torch.load(file_path) 
        mfp = np.zeros(self.out_dim)
        for frag_info in fragment_infos:
            if frag_info in self.bitInfos_to_fp_index_map:
                mfp[self.bitInfos_to_fp_index_map[frag_info]] = 1
        return torch.tensor(mfp).float()
    
    def build_rankingset(self, split, predefined_FP=None):
        return super().build_rankingset(split, predefined_FP)

    def build_mfp_for_new_SMILES(self, smiles):
        from notebook_and_scripts.SMILES_fragmenting.build_dataset_specific_FP.find_frags import count_circular_substructures
        mfp = np.zeros(self.out_dim)
        
        bitInfos_with_count = count_circular_substructures(smiles)
        for bitInfo in bitInfos_with_count:
            if bitInfo in self.bitInfos_to_fp_index_map:
                mfp[self.bitInfos_to_fp_index_map[bitInfo]] = 1
        return torch.tensor(mfp).float()
    
    def build_inference_ranking_set_with_everything(self, fp_dim, max_radius, use_hyun_fp = False, test_on_deepsat_retrieval_set = False):
        if use_hyun_fp:
            rankingset_path = "/root/gurusmart/MorganFP_prediction/inference_data/inference_rankingset_with_stable_sort/hyun_fp_stacked_together_sparse/FP_normalized.pt" # FP and FP_normalized are the same
        else:
            rankingset_path = f"/root/gurusmart/MorganFP_prediction/inference_data/inference_rankingset_with_stable_sort/non_collision_FP_rankingset_max_radius_{max_radius}_dim_{fp_dim}_stacked_together/FP.pt"
        if test_on_deepsat_retrieval_set:
            rankingset_path = rankingset_path.replace("FP_normalized", "FP_normalized_deepsat_retrieval_set")
        print(f"loading {rankingset_path}")
        return torch.load(rankingset_path)#.to("cuda")
        
class FP_Loader_Configer():
    def __init__(self):
        # print("FP_Loader_Configer is initialized\n\n\n\n")
        self.fp_loader = None
    
    def select_version(self, version):
        if version == "MFP_Specific_Radius":
            print("choosing Specific_Radius_MFP_loader")
            self.fp_loader = Specific_Radius_MFP_loader()
        elif version == "DB_Specific":
            print("choosing DB_Specific_FP_loader")
            self.fp_loader = DB_Specific_FP_loader()
        elif version == "Hash_Entropy":
            print("choosing Hash_Entropy_FP_loader")
            self.fp_loader = Hash_Entropy_FP_loader()

        else:
            raise ValueError("version should be either Specific_Radius or DB_Specific")

fp_loader_configer = FP_Loader_Configer()