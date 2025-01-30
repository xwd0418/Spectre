import torch, os, heapq
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import Draw


def unpack_inputs(inputs):
    # input shape: 1* (n, 3)
    for i, vals in enumerate(inputs[0]):
        # if vals is [-1, -1, -1]
        if vals[0]==-1 and vals[1]==-1 and vals[2]==-1:
            hsqc_start=i+1
        elif vals[0]==-2 and vals[1]==-2 and vals[2]==-2:
            hsqc_end=i
        elif vals[0]==-3 and vals[1]==-3 and vals[2]==-3:
            c_nmr_start=i+1
        elif vals[0]==-4 and vals[1]==-4 and vals[2]==-4:
            c_nmr_end=i
        elif vals[0]==-5 and vals[1]==-5 and vals[2]==-5:
            h_nmr_start=i+1
        elif vals[0]==-6 and vals[1]==-6 and vals[2]==-6:
            h_nmr_end=i
            
    hsqc = inputs[0,hsqc_start:hsqc_end]
    c_tensor = inputs[0,c_nmr_start:c_nmr_end,0]
    h_tensor = inputs[0,h_nmr_start:h_nmr_end,1]
    return hsqc, c_tensor, h_tensor

def unpack_inputs_no_delimiter(inputs, NMR_type_indicator):
    # input shape: (n, 3)
    # indicator shape: somehting like tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3])
    
    unique_values, indices = torch.unique_consecutive(NMR_type_indicator, return_inverse=True)

    # Compute start and end positions
    indices_location = {}
    for value in unique_values:
        positions = (NMR_type_indicator == value).nonzero(as_tuple=True)[0]
        indices_location[int(value)] = (positions[0].item(), positions[-1].item()+1)
    
    hsqc_start, hsqc_end = indices_location[0]
    c_nmr_start, c_nmr_end = indices_location[1]
    h_nmr_start, h_nmr_end = indices_location[2]
    
    hsqc = inputs[hsqc_start:hsqc_end]
    c_tensor = inputs[c_nmr_start:c_nmr_end,0]
    h_tensor = inputs[h_nmr_start:h_nmr_end,1]
    return hsqc, c_tensor, h_tensor



def retrieve_top_k_by_dir(dir, prediction_query, smiles_and_names , k=30):
    query = F.normalize(prediction_query, dim=1, p=2.0).squeeze()
    results = []
    for  ranker_f in sorted(os.listdir(dir), key=lambda x:int(x.split("_")[1].split(".")[0])):
    # for num_ranker_data, ranker_f in [(54,"FP_54.pt")]:
        # print(ranker_f)
        num_ranker_data = int(ranker_f.split("_")[1].split(".")[0])
        data = torch.load(os.path.join(dir, ranker_f)).to("cuda")
        query_products = (data @ query)
        values, indices = torch.topk(query_products,k=k)
        if len(results) == 0:
            for value, idx in zip(values, indices):
                real_idx = idx + 2000*num_ranker_data
                heapq.heappush(results, (value, real_idx, data[idx].nonzero()))
        else:
            for value, idx in zip(values, indices):
                real_idx = idx + 2000*num_ranker_data
                heapq.heappushpop(results, (value, real_idx, data[idx].nonzero()))    
                
                        
    results.sort(key=lambda x: x[0],reverse=True)
    ret = [(value, smiles_and_names[i], fp) for value, i, fp in results]
    # print(torch.tensor(idx))
    # retrieved_FP = [all_fp[i] for i in idx]
    # print(results[0])
  
    return ret

def retrieve_top_k_by_rankingset(data, prediction_query, smiles_and_names , k=30):
    query = F.normalize(prediction_query, dim=1, p=2.0).squeeze()

    results = []
    query_products = (data @ query)
    values, indices = torch.topk(query_products,k=k)
    
    for value, idx in zip(values, indices):
        results.append((value, idx, data[idx]))
                
                        
    results.sort(key=lambda x: x[0],reverse=True)
    ret = [(value, smiles_and_names[i], fp) for value, i, fp in results]
    # print(torch.tensor(idx))
    # retrieved_FP = [all_fp[i] for i in idx]
    # print(results[0])
  
    return ret

def compute_cos_sim(fp1, fp2):
    return (fp1 @ fp2) / (torch.norm(fp1) * torch.norm(fp2)).item()

def show_retrieved_mol_with_highlighted_frags(retrieved_mol_smiles, predicted_FP, FP_index_to_frags_mapping, fp_gen, ao):
    predicted_frags = [FP_index_to_frags_mapping[i.item()] for i in predicted_FP.nonzero()]
    mol = Chem.MolFromSmiles(retrieved_mol_smiles)
    if mol is None:
        print(f"Failed to parse {retrieved_mol_smiles}")
        raise ValueError(f"Failed to parse {retrieved_mol_smiles}")
    mol = Chem.AddHs(mol)

    # Compute Morgan fingerprint with radius 
    fp = fp_gen.GetFingerprint(mol, additionalOutput=ao)
    info = ao.GetBitInfoMap()

    # Extract circular subgraphs

    highlight_bonds = set()
    # display(info)
    for bit_id, atom_envs in info.items():
        for atom_idx, curr_radius in atom_envs:
            # Get the circular environment as a subgraph
            env = Chem.FindAtomEnvironmentOfRadiusN(mol, curr_radius, atom_idx)
            submol = Chem.PathToSubmol(mol, env)
            frag_smiles = Chem.MolToSmiles(submol, canonical=True)
            
            if frag_smiles and frag_smiles in predicted_frags:
                highlight_bonds.update(env)
                
    highlight_bonds = list(highlight_bonds)

    # Visualize with highlights
    img = Draw.MolToImage(mol, highlightBonds=highlight_bonds, size=(600,600))
    # img.show()
    return img
                    
        