import numpy as np, pickle
import torch
from torch.nn.utils.rnn import pad_sequence


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
    probability = data/(total_dataset_size)
    if use_natural_log:
        entropy = (probability * np.log(np.clip(probability,1e-7 ,1)) )
    else:
        entropy = (probability * np.log2(np.clip(probability,1e-7 ,1)) )
    return entropy

def keep_smallest_entropy(data, total_dataset_size, size=6144,  use_natural_log=False):
    entropy = compute_entropy(data, total_dataset_size, use_natural_log)
    indices_of_min_6144 = np.argsort(entropy)[:size]
    # print(entropy, indices_of_min_6144)
    total_entropy = entropy[indices_of_min_6144].sum()
    return total_entropy, indices_of_min_6144

''' 
I have a pre-built 15*6144 fp 
now I select top 6144 bits of highest entropy, from r0 up to r(x)
'''
def convert_bits_positions_to_array(FP_on_bits, length):
    FP_on_bits= FP_on_bits.astype(int)
    out = np.zeros(length)
    out[FP_on_bits] = 1
    return out

class Specific_Radius_MFP_loader():
    def __init__(self) -> None:
        self.path_pickles = '/root/MorganFP_prediction/reproduce_previous_works/smart4.5/notebooks/dataset_building/FP_on_bits_pickles/'
        
    def setup(self, only_2d = False):
        self.train_1d = pickle.load(open(self.path_pickles + 'Normal_FP_on_bits_r0_r15_len_6144_1d_train.pkl', 'rb'))
        self.train_2d = pickle.load(open(self.path_pickles + 'Normal_FP_on_bits_r0_r15_len_6144_2d_train.pkl', 'rb'))
        
        self.count = np.zeros(6144*16)
        if not only_2d:
            for FP_on_bits in self.train_1d.values():
                self.count += convert_bits_positions_to_array(FP_on_bits, 6144*16)
        for FP_on_bits in self.train_2d.values():
            self.count += convert_bits_positions_to_array(FP_on_bits, 6144*16)
  
        # also val and test
        self.val_1d = pickle.load(open(self.path_pickles + 'Normal_FP_on_bits_r0_r15_len_6144_1d_val.pkl', 'rb'))
        self.test_1d = pickle.load(open(self.path_pickles + 'Normal_FP_on_bits_r0_r15_len_6144_1d_test.pkl', 'rb'))
        self.val_2d = pickle.load(open(self.path_pickles + 'Normal_FP_on_bits_r0_r15_len_6144_2d_val.pkl', 'rb'))
        self.test_2d = pickle.load(open(self.path_pickles + 'Normal_FP_on_bits_r0_r15_len_6144_2d_test.pkl', 'rb'))
       
    
    def set_max_radius(self, max_radius, only_2d = False):
        self.max_radius = max_radius
        count_for_current_radius = self.count[:6144*(max_radius+1)]
        if only_2d:
            total_dataset_size = len(self.train_2d)
        else:
            total_dataset_size = len(self.train_1d) + len(self.train_2d)
        __entropy, self.indices_kept = keep_smallest_entropy(count_for_current_radius, total_dataset_size)
        # self.indices_kept = list(self.indices_kept)
        assert len(self.indices_kept) == 6144, "should keep only 6144 highest entropy bits"
        
    def build_mfp(self, file_idx, dataset, split):
        if dataset == '1d':
            if split == 'train':
                FP_on_bits = self.train_1d[file_idx]
            elif split == 'val':
                FP_on_bits = self.val_1d[file_idx]
            elif split == 'test':
                FP_on_bits = self.test_1d[file_idx]
        elif dataset == '2d':
            if split == 'train':
                FP_on_bits = self.train_2d[file_idx]
            elif split == 'val':
                FP_on_bits = self.val_2d[file_idx]
            elif split == 'test':
                FP_on_bits = self.test_2d[file_idx]
        else:
            raise ValueError("dataset should be either 1d or 2d")
        
        mfp = convert_bits_positions_to_array(FP_on_bits, 6144*16)
        mfp = mfp[self.indices_kept]
        assert(len(mfp) == 6144)
        return torch.tensor(mfp).float()
    
    def build_rankingset(self, dataset, split):
        if dataset == '1d':
            if split == 'val':
                FP_on_bits_mappings = self.val_1d
            elif split == 'test':
                FP_on_bits_mappings = self.test_1d
        elif dataset == '2d':      
            if split == 'val':
                FP_on_bits_mappings = self.val_2d
            elif split == 'test':
                FP_on_bits_mappings = self.test_2d
         
        if dataset == '1d' or dataset == '2d':   
            files = [self.build_mfp(file_idx, dataset, split) for file_idx in FP_on_bits_mappings.keys()]             
        elif dataset == "both":
            if split == 'val':
                files  = [self.build_mfp(file_idx, '1d', split) for file_idx in self.val_1d.keys()]
                files += [self.build_mfp(file_idx, '2d', split) for file_idx in self.val_2d.keys()]
            elif split == 'test':
                files  = [self.build_mfp(file_idx, '1d', split) for file_idx in self.test_1d.keys()]
                files += [self.build_mfp(file_idx, '2d', split) for file_idx in self.test_2d.keys()]
                
        out = torch.vstack(files)
        return out
    
    

specific_radius_mfp_loader = Specific_Radius_MFP_loader()

# def s(dataset, file_index, fp_suffix):
                
    