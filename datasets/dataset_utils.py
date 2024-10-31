import numpy as np, pickle
import torch
from torch.nn.utils.rnn import pad_sequence
from rdkit.Chem import rdMolDescriptors
from rdkit import Chem
import sys, pathlib
repo_path = pathlib.Path(__file__).resolve().parents[1]

def pad(sequence):
  """
    Assume sequence is a (batch)list of sequences that can be variable size.

    Returns {
      sequence: tensor of padded sequence
      padding_mask: binary tensor marking padding
    }
  """
  sequence = pad_sequence([
      torch.tensor(v, dtype=torch.float32) if type(v) is not torch.Tensor
      else v.type(torch.float32)
      for v in sequence
  ], batch_first=True)
  return sequence

def pad_and_mask(sequence):
  """
    Assume sequence is a (batch)list of sequences that can be variable size.

    Returns {
      sequence: tensor of padded sequence
      padding_mask: binary tensor marking padding
    }
  """
  sequence = pad_sequence([
      torch.tensor(v, dtype=torch.float32) if type(v) is not torch.Tensor
      else v.type(torch.float32)
      for v in sequence
  ], batch_first=True)
  padding_mask = ~torch.any((sequence > 0), dim=2)
  return {
      "sequence": sequence,
      "padding_mask": padding_mask
  }

def tokenise_and_mask(smiles, tokeniser):
  """
    Assume sequence is [batch] sized list of SMILES strings

    Returns {
      raw_smiles: the original list of smiles passed in
      target: right-shifted padded list of token ids (b_s, seq_len)
      target_mask: right-shifted padding mask
      decoder_inputs: tokenised inputs (minus end(&) token)
      decoder_mask: padding mask
    }
  """

  obj = tokeniser.tokenise(smiles, pad=True)

  out_tokens = obj["original_tokens"]
  out_mask = obj["original_pad_masks"]

  token_ids = torch.tensor(
      tokeniser.convert_tokens_to_ids(out_tokens), dtype=torch.int64)
  padding_mask = torch.tensor(out_mask, dtype=torch.bool)
  targets = token_ids.clone()[:, 1:]
  target_mask = padding_mask.clone()[:, 1:]
  decoder_inputs = token_ids[:, :-1]
  decoder_mask = padding_mask[:, :-1]
  return {
      "raw_smiles": smiles,
      "target": targets,
      "target_mask": target_mask,
      "decoder_inputs": decoder_inputs,
      "decoder_mask": decoder_mask
  }

def tokenise_and_mask_encoder(smiles, tokeniser):
  """
    Assume sequence is [batch] sized list of SMILES strings

    Returns {
      raw_smiles: the original list of smiles passed in
      target: right-shifted padded list of token ids (b_s, seq_len)
      target_mask: right-shifted padding mask
      decoder_inputs: tokenised inputs (minus end(&) token)
      decoder_mask: padding mask
    }
  """

  obj = tokeniser.tokenise(smiles, pad=True)

  out_tokens = obj["original_tokens"]
  out_mask = obj["original_pad_masks"]

  token_ids = torch.tensor(
      tokeniser.convert_tokens_to_ids(out_tokens), dtype=torch.int64)
  padding_mask = torch.tensor(out_mask, dtype=torch.bool)

  # (b_s, seq_len)
  decoder_inputs = token_ids[:, :-1]
  decoder_mask = padding_mask[:, :-1]
  return {
      "raw_smiles": smiles,
      "encoder_inputs": decoder_inputs,
      "encoder_mask": decoder_mask
  }



# utils about entropy and MFP
def compute_entropy(data, total_dataset_size, use_natural_log=False):
    # data: an array of counts of each bit, by default in the size of self.out_dim*(max_radius+1)
    probability = data/(total_dataset_size)
    if use_natural_log:
        entropy = (probability * np.log(np.clip(probability,1e-7 ,1)) )
    else:
        entropy = (probability * np.log2(np.clip(probability,1e-7 ,1)) )
    return entropy

def keep_smallest_entropy(data, total_dataset_size, size,  use_natural_log=False):
    entropy = compute_entropy(data, total_dataset_size, use_natural_log)
    indices_of_min_6144 = np.argsort(entropy)[:size]
    # print(entropy, indices_of_min_6144)
    total_entropy = entropy[indices_of_min_6144].sum()
    return total_entropy, indices_of_min_6144

''' 
convert the on bit positions to an array of 0s and 1s
'''
def convert_bits_positions_to_array(FP_on_bits, length):
    FP_on_bits= FP_on_bits.astype(int)
    out = np.zeros(length)
    out[FP_on_bits] = 1
    return out

def generate_normal_FP_on_bits(mol, radius=2, length=6144):
    # Dictionary to store information about which substructures contribute to setting which bits
    bitInfo = {}
    # Generate the fingerprint with bitInfo to track the substructures contributing to each bit
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=length, bitInfo=bitInfo)
    on_bits = np.array(fp.GetOnBits())
    return on_bits

class Specific_Radius_MFP_loader():
    def __init__(self, ) -> None:
        self.path_pickles = f'{repo_path}/notebooks/dataset_building/FP_on_bits_pickles/'
        
    def setup(self, only_2d = False, FP_building_type = "Normal", out_dim=6144):
        assert (FP_building_type in ["Normal", "Exact"]), "Only Normal/Exact FP is supported"
        self.out_dim = out_dim
        self.single_FP_size = 6144 if FP_building_type == "Normal" else 1024
        self.train_1d = pickle.load(open(self.path_pickles + f'{FP_building_type}_FP_on_bits_r0_r15_len_{self.single_FP_size}_1d_train.pkl', 'rb'))
        self.train_2d = pickle.load(open(self.path_pickles + f'{FP_building_type}_FP_on_bits_r0_r15_len_{self.single_FP_size}_2d_train.pkl', 'rb'))
        
        self.count = np.zeros(self.single_FP_size*16)
        if not only_2d:
            for FP_on_bits in self.train_1d.values():
                self.count += convert_bits_positions_to_array(FP_on_bits, self.single_FP_size*16)
        for FP_on_bits in self.train_2d.values():
            self.count += convert_bits_positions_to_array(FP_on_bits, self.single_FP_size*16)
  
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
    
    def build_rankingset(self, split):         
        path_to_load_full_info_indices = f"{repo_path}/datasets/{split}_indices_of_full_info_NMRs.pkl"
        file_idx_for_ranking_set = pickle.load(open(path_to_load_full_info_indices, "rb"))
        files  = [self.build_mfp(int(file_idx.split(".")[0]), "2d", split) for file_idx in sorted(file_idx_for_ranking_set)]
             
        out = torch.vstack(files)
        return out
    
    # def build_inference_ranking_set_with_everything(self):
        # train_files_2d = [self.build_mfp(file_idx, "2d", "train") for file_idx in range(len(self.train_2d))]
        # val_files_2d = [self.build_mfp(file_idx, "2d", "val") for file_idx in range(len(self.val_2d))]
        # test_files_2d = [self.build_mfp(file_idx, "2d", "test") for file_idx in range(len(self.test_2d))]
        # train_files_1d = [self.build_mfp(file_idx, "1d", "train") for file_idx in range(len(self.train_1d))]
        # val_files_1d = [self.build_mfp(file_idx, "1d", "val") for file_idx in range(len(self.val_1d))]
        # test_files_1d = [self.build_mfp(file_idx, "1d", "test") for file_idx in range(len(self.test_1d))]
                         
        # out = torch.vstack(train_files_2d + val_files_2d + test_files_2d + train_files_1d + val_files_1d + test_files_1d)
        # return torch.vstack(train_files_2d)
    def build_mfp_for_new_SMILES(self, smiles):
        num_plain_FPs = 16 # radius from 0 to 15
        
        mol = Chem.MolFromSmiles(smiles)
        mol_H = Chem.AddHs(mol) # add implicit Hs to the molecule
        all_plain_fps_on_bits = []
        for radius in range(num_plain_FPs):
            all_plain_fps_on_bits.append(generate_normal_FP_on_bits(mol_H, radius=radius, length=self.single_FP_size) + radius*self.single_FP_size)
        FP_on_bits = np.concatenate(all_plain_fps_on_bits)
        
        mfp = convert_bits_positions_to_array(FP_on_bits, self.single_FP_size*16)
        mfp = mfp[self.indices_kept]
        return torch.tensor(mfp).float()
    
specific_radius_mfp_loader = Specific_Radius_MFP_loader()

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot_NMR(hsqc, c_tensor, h_tensor):
    # print(hsqc, c_tensor, h_tensor)
    # Create a 2x2 grid for subplots
    fig = plt.figure(figsize=(6, 4.8))  # Overall figure size
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 20], width_ratios=[1, 20])

    # Create subplots in different locations and sizes
    ax1 = fig.add_subplot(gs[1, 1])  # Takes up the first row
    if hsqc is not None:
        pos = hsqc[hsqc[:,2]>0]
        neg = hsqc[hsqc[:,2]<0]
        ax1.scatter(pos[:,1], pos[:,0], c="red", label="1 or 3 H bond", s=5)
        ax1.scatter(neg[:,1], neg[:,0], c="blue", label="2 H bond", s=5)
        # print("scatter!!")
        # print(pos, neg)
    ax1.set_title("HSQC")
    ax1.set_xlabel('Proton Shift (H)')  # X-axis label
    ax1.set_xlim([0, 7.5])
    ax1.set_ylim([0, 180])
    ax1.invert_yaxis()
    ax1.invert_xaxis()
    ax1.legend()


    ax2 = fig.add_subplot(gs[1, 0])  # Smaller subplot
    if c_tensor is not None:
        ax2.scatter( torch.ones(len(c_tensor)), c_tensor, c="black", s=2)
    ax2.set_ylim([0, 180])
    ax2.set_title("C-NMR")
    ax2.set_ylabel('Carbon Shift (C)')
    ax2.set_xticks([])
    ax2.invert_yaxis()
    ax2.invert_xaxis()

    ax3 = fig.add_subplot(gs[0, 1])  # Smaller subplot
    if h_tensor is not None:
        ax3.scatter(h_tensor, torch.ones(len(h_tensor)),c="black", s=2)
    ax3.set_xlim([0, 7.5])
    ax3.set_title("H-NMR")
    ax3.set_yticks([])
    ax3.invert_yaxis()
    ax3.invert_xaxis()

    # Adjust layout to prevent overlapping
    plt.tight_layout()

    # Show the plot
    plt.show()

# def s(dataset, file_index, fp_suffix):
if __name__ == "__main__":
    specific_radius_mfp_loader.setup(only_2d = False, FP_building_type = "Normal")
    specific_radius_mfp_loader.set_max_radius(2, only_2d = False)
    
    for file_idx in range(len(specific_radius_mfp_loader.train_2d)):
        mfp = specific_radius_mfp_loader.build_mfp(file_idx, "2d", "train")
        print(mfp)
        break
    # train_files_2d = [self.build_mfp(file_idx, "2d", "train") )]
        # val_files_2d = [self.build_mfp(file_idx, "2d", "val") for file_idx in range(len(self.val_2d))]
        # test_files_2d = [self.build_mfp(file_idx, "2d", "test") for file_idx in range(len(self.test_2d))]
        # train_files_1d = [self.build_mfp(file_idx, "1d", "train") for file_idx in range(len(self.train_1d))]
        # val_files_1d = [self.build_mfp(file_idx, "1d", "val") for file_idx in range(len(self.val_1d))]
        # test_files_1d = [self.build_mfp(file_idx, "1d", "test") for file_idx in range(len(self.test_1d))]
        
    print("Done")
                
    