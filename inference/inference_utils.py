import torch, os, heapq
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import Draw
import numpy as np
import sys, pathlib, json, yaml
from pathlib import Path

### model selection ###

# stable sort
# def find_checkpoint_path(model_type):
#     match model_type:
#         case "C-NMR":
#             checkpoint_path = Path("/root/gurusmart/MorganFP_prediction/reproduce_previous_works/stable_argsort/train_on_all_data_possible/only_c_trial_1/checkpoints/epoch=28-step=46864.ckpt")
#         case "H-NMR":
#             checkpoint_path = Path("/root/gurusmart/MorganFP_prediction/reproduce_previous_works/stable_argsort/train_on_all_data_possible/only_h_trial_1/checkpoints/epoch=23-step=35016.ckpt")
#         case "HSQC":
#             checkpoint_path = Path("/root/gurusmart/MorganFP_prediction/reproduce_previous_works/stable_argsort/train_on_all_data_possible/only_hsqc_trial_1/checkpoints/epoch=21-step=37708.ckpt")
#         case "only_1d":
#             checkpoint_path = Path("/root/gurusmart/MorganFP_prediction/reproduce_previous_works/stable_argsort/train_on_all_data_possible/only_1d_trial_1/checkpoints/epoch=26-step=39393.ckpt")
#         case "All-NMR":
#             checkpoint_path = Path("/root/gurusmart/MorganFP_prediction/reproduce_previous_works/stable_argsort/flexible_models_best_FP/r0_r2_FP_trial_1/checkpoints/epoch=32-all_inputs.ckpt")
#         case "HSQC_C-NMR":
#             checkpoint_path = Path("/root/gurusmart/MorganFP_prediction/reproduce_previous_works/stable_argsort/flexible_models_best_FP/r0_r2_FP_trial_1/checkpoints/epoch=32-HSQC_C_NMR.ckpt")
#         case "HSQC_H-NMR":
#             checkpoint_path = Path("/root/gurusmart/MorganFP_prediction/reproduce_previous_works/stable_argsort/flexible_models_best_FP/r0_r2_FP_trial_1/checkpoints/epoch=46-HSQC_H_NMR.ckpt")
#         case "only_1d_DTD":
#             checkpoint_path = Path("/root/gurusmart/MorganFP_prediction/reproduce_previous_works/stable_argsort/flexible_models_best_FP/r0_r2_FP_trial_1/checkpoints/epoch=27-only_1d.ckpt")
#         case "only_C-NMR_DTD":
#             checkpoint_path = Path("/root/gurusmart/MorganFP_prediction/reproduce_previous_works/stable_argsort/flexible_models_best_FP/r0_r2_FP_trial_1/checkpoints/epoch=45-only_C_NMR.ckpt")
#         case _:
#             raise ValueError(f"model_type: {model_type} not recognized")
        
#     return checkpoint_path

# # stop on cosine
# def find_checkpoint_path(model_type):
#     match model_type:
#         # case "C-NMR":
#         #     checkpoint_path = Path("/root/gurusmart/MorganFP_prediction/reproduce_previous_works/stable_argsort/train_on_all_data_possible/only_c_trial_1/checkpoints/epoch=28-step=46864.ckpt")
#         # case "H-NMR":
#         #     checkpoint_path = Path("/root/gurusmart/MorganFP_prediction/reproduce_previous_works/stable_argsort/train_on_all_data_possible/only_h_trial_1/checkpoints/epoch=23-step=35016.ckpt")
#         # case "HSQC":
#         #     checkpoint_path = Path("/root/gurusmart/MorganFP_prediction/reproduce_previous_works/stable_argsort/train_on_all_data_possible/only_hsqc_trial_1/checkpoints/epoch=21-step=37708.ckpt")
#         # case "only_1d":
#         #     checkpoint_path = Path("/root/gurusmart/MorganFP_prediction/reproduce_previous_works/stable_argsort/train_on_all_data_possible/only_1d_trial_1/checkpoints/epoch=26-step=39393.ckpt")
#         case "All-NMR":
#             checkpoint_path = Path("/root/gurusmart/MorganFP_prediction/reproduce_previous_works/stop_on_cosine/flexible_models_larger_no_jittering/r0_r2_FP_trial_1/checkpoints/epoch=28-all_inputs.ckpt")
#         # case "HSQC_C-NMR":
#         #     checkpoint_path = Path("/root/gurusmart/MorganFP_prediction/reproduce_previous_works/stable_argsort/flexible_models_best_FP/r0_r2_FP_trial_1/checkpoints/epoch=32-HSQC_C_NMR.ckpt")
#         # case "HSQC_H-NMR":
#         #     checkpoint_path = Path("/root/gurusmart/MorganFP_prediction/reproduce_previous_works/stable_argsort/flexible_models_best_FP/r0_r2_FP_trial_1/checkpoints/epoch=46-HSQC_H_NMR.ckpt")
#         # case "only_1d_DTD":
#         #     checkpoint_path = Path("/root/gurusmart/MorganFP_prediction/reproduce_previous_works/stable_argsort/flexible_models_best_FP/r0_r2_FP_trial_1/checkpoints/epoch=27-only_1d.ckpt")
#         # case "only_C-NMR_DTD":
#         #     checkpoint_path = Path("/root/gurusmart/MorganFP_prediction/reproduce_previous_works/stable_argsort/flexible_models_best_FP/r0_r2_FP_trial_1/checkpoints/epoch=45-only_C_NMR.ckpt")
#         case _:
#             raise ValueError(f"model_type: {model_type} not recognized")
        
#     return checkpoint_path

# larger model, jittering 
def find_checkpoint_path(model_type):
    match model_type:
        case "C-NMR":
            checkpoint_path = Path("/root/gurusmart/MorganFP_prediction/reproduce_previous_works/stop_on_cosine/all_data_possible/only_c_trial_1/checkpoints/epoch=97-step=79184.ckpt")
        case "H-NMR":
            checkpoint_path = Path("/root/gurusmart/MorganFP_prediction/reproduce_previous_works/stop_on_cosine/all_data_possible/only_h_trial_1/checkpoints/epoch=86-step=63510.ckpt")
        case "HSQC":
            checkpoint_path = Path("/root/gurusmart/MorganFP_prediction/reproduce_previous_works/stop_on_cosine/all_data_possible/only_hsqc_trial_1/checkpoints/epoch=66-step=57419.ckpt")
        case "only_1d":
            checkpoint_path = Path("/root/gurusmart/MorganFP_prediction/reproduce_previous_works/stop_on_cosine/all_data_possible/only_1d_trial_1/checkpoints/epoch=66-step=48910.ckpt")
        case "All-NMR":
            checkpoint_path = Path("/root/gurusmart/MorganFP_prediction/reproduce_previous_works/stop_on_cosine/larger_flexible_models_3072dim/r0_r4_FP_trial_1/checkpoints/epoch=86-all_inputs.ckpt")
        case "HSQC_C-NMR":
            checkpoint_path = Path("/root/gurusmart/MorganFP_prediction/reproduce_previous_works/stop_on_cosine/larger_flexible_models_3072dim/r0_r4_FP_trial_1/checkpoints/epoch=91-HSQC_C_NMR.ckpt")
        case "HSQC_H-NMR":
            checkpoint_path = Path("/root/gurusmart/MorganFP_prediction/reproduce_previous_works/stop_on_cosine/larger_flexible_models_3072dim/r0_r4_FP_trial_1/checkpoints/epoch=87-HSQC_H_NMR.ckpt")
       
        # case "only_1d_DTD":
        #     checkpoint_path = Path("/root/gurusmart/MorganFP_prediction/reproduce_previous_works/stable_argsort/flexible_models_best_FP/r0_r2_FP_trial_1/checkpoints/epoch=27-only_1d.ckpt")
        # case "only_C-NMR_DTD":
        #     checkpoint_path = Path("/root/gurusmart/MorganFP_prediction/reproduce_previous_works/stable_argsort/flexible_models_best_FP/r0_r2_FP_trial_1/checkpoints/epoch=45-only_C_NMR.ckpt")
        case _:
            raise ValueError(f"model_type: {model_type} not recognized")
        
    return checkpoint_path


# get model and dataloader
from models.optional_input_ranked_transformer import OptionalInputRankedTransformer
from datasets.optional_2d_folder_dataset import OptionalInputDataModule
from datasets.dataset_utils import fp_loader_configer
specific_radius_mfp_loader = fp_loader_configer.fp_loader

def choose_model(model_type, include_test_loader=True):
    
    checkpoint_path = find_checkpoint_path(model_type)
    
    model_path = checkpoint_path.parents[1]
    hyperpaerameters_path = model_path / "hparams.yaml"

    # checkpoint_path = model_path / "checkpoints/epoch=14-step=43515.ckpt"


    with open(hyperpaerameters_path, 'r') as file:
        hparams = yaml.safe_load(file)
        
    FP_building_type = hparams['FP_building_type'].split("_")[-1]
    only_2d = not hparams['use_oneD_NMR_no_solvent']
    # only_2d = True
    print(FP_building_type)
    max_radius = int(hparams['FP_choice'].split("_")[-1][1:])
    print("max_radius: ", max_radius)
    
    if  max_radius!=specific_radius_mfp_loader.max_radius or only_2d!=specific_radius_mfp_loader.only_2d:
        specific_radius_mfp_loader.setup(only_2d=only_2d,FP_building_type=FP_building_type)
        specific_radius_mfp_loader.set_max_radius(int(hparams['FP_choice'].split("_")[-1][1:]), only_2d=only_2d)


    del hparams['checkpoint_path'] # prevent double defition of checkpoint_path
    hparams['use_peak_values'] = False
    hparams['num_workers'] = 0
    model = OptionalInputRankedTransformer.load_from_checkpoint(checkpoint_path, **hparams)
    
    if not include_test_loader:
        return hparams, model
    datamodule = OptionalInputDataModule(dir="/workspace/SMILES_dataset", FP_choice=hparams["FP_choice"], input_src=["HSQC", "oneD_NMR"], batch_size=1, parser_args=hparams)
    datamodule.setup("predict")
    loader_all_inputs, loader_HSQC_H_NMR, loader_HSQC_C_NMR, loader_only_hsqc, loader_only_1d, loader_only_H_NMR, loader_only_C_NMR = \
        datamodule.predict_dataloader()
        
    match model_type:
        case "C-NMR":
            test_loader = loader_only_C_NMR
        case "H-NMR":
            test_loader = loader_only_H_NMR
        case "HSQC":
            test_loader = loader_only_hsqc
        case "All-NMR":
            test_loader = loader_all_inputs  
        case "only_1d":
            test_loader = loader_only_1d
        case "HSQC_C-NMR":
            test_loader = loader_HSQC_C_NMR
        case "HSQC_H-NMR":
            test_loader = loader_HSQC_H_NMR
        case "only_1d_DTD":
            test_loader = loader_only_1d
        case "only_C-NMR_DTD":
            test_loader = loader_only_C_NMR
            
                          
        case _:
            raise ValueError(f"model_type: {model_type} not recognized")
    model.eval()
    return hparams, model, test_loader




### input transformation ###
'''
deprecated. used for the old delimiter model
'''
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

### retrieval ###

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

def retrieve_top_k_by_rankingset(data, prediction_query, smiles_and_names, k=30, filter_by_MW=None):
    # data means the rankingset data
    # Expect filter by MW to be [lower_bnound, upper_bound]
    query = F.normalize(prediction_query, dim=1, p=2.0).squeeze()

    results = []
    query_products = (data @ query)
    if filter_by_MW is None:
        values, indices = torch.topk(query_products,k=k)
        for value, idx in zip(values, indices):
            results.append((value, smiles_and_names[idx], data[idx]))
    else:
        values, indices = torch.topk(query_products,k=1000)
        for value, idx in zip(values, indices):
            if filter_by_MW[0] <= smiles_and_names[idx][2] <= filter_by_MW[1] :
                results.append((value, smiles_and_names[idx], data[idx]))
                if len(results) == k:
                    break
        
    
    
    return results        
                        

def compute_cos_sim(fp1, fp2):
    return (fp1 @ fp2) / (torch.norm(fp1) * torch.norm(fp2)).item()



def show_retrieved_mol_with_highlighted_frags(retrieved_mol_smiles, predicted_FP, FP_index_to_frags_mapping, fp_gen, ao):
    '''
    used in db-specific FP
    This functions visualizes the retrieved molecule with the predicted fragments highlighted
    '''
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
                    
    
######  for unknow compound ######    
def build_input(compound_dir, mode = None, include_hsqc = True, include_c_nmr = True, include_h_nmr = True, include_MW = True):
    """
    build input tensor for unkonw compound
    mode is either None, "no_sign", "flip_sign"
    
    return inputs, NMR_type_indicator
    """
    print("\n\n")
    print(compound_dir.split("/")[-1])
    print("\n")
    def load_2d():
        return torch.tensor(np.loadtxt(os.path.join(compound_dir, "HSQC.txt"), delimiter=",")).float()
    def load_1d(nmr ):
        vals = np.loadtxt(os.path.join(compound_dir, f"{nmr}.txt"), delimiter=",")
        vals = torch.tensor(np.unique(vals)).float()
        if nmr == "H":
            return F.pad(vals.view(-1, 1), (1, 1), "constant", 0)
        elif nmr == "C":
            return F.pad(vals.view(-1, 1), (0, 2), "constant", 0)
        else:
            raise ValueError("nmr_type must be either H or C")
    
    hsqc = load_2d()
    if hsqc is not None:
        hsqc[:,[0,1]] = hsqc[:,[1,0]]
        if mode == "no_sign":
            hsqc = torch.abs(hsqc)
        elif mode == "flip_sign":
            hsqc[:,2] = -hsqc[:,2]
    c_tensor = load_1d("C")
    h_tensor = load_1d("H")
    input_NMRs = []
    NMR_type_indicator = []
    if include_hsqc:
        input_NMRs.append(hsqc)
        NMR_type_indicator+= [0]*hsqc.shape[0]
    if include_c_nmr:
        input_NMRs.append(c_tensor)
        NMR_type_indicator+= [1]*c_tensor.shape[0]
    if include_h_nmr:
        input_NMRs.append(h_tensor)
        NMR_type_indicator+= [2]*h_tensor.shape[0]
    inputs = torch.vstack(input_NMRs)   
    if include_MW:
        with open(os.path.join(compound_dir, "mw.txt"), 'r') as file:
            # Read the content of the file
            content = file.read()
            # Convert the content to a float
            mw = float(content)
        mol_weight = torch.tensor([mw,0,0]).float()
        inputs = torch.vstack([inputs, mol_weight])
        input_NMRs.append(mol_weight)
        NMR_type_indicator.append(3)
    NMR_type_indicator = torch.tensor(NMR_type_indicator)
    
    # print(inputs)
    # print(hsqc, c_tensor, h_tensor)
    # plot_NMR(hsqc, c_tensor[:,0], h_tensor[:,0])
    return inputs, NMR_type_indicator


def inference_topK(inputs, NMR_type_indicator, model, rankingset_data, smiles_and_names, 
                   k=5, mode = None, ground_truth_FP=None,
                   fp_type = "MFP_Specific_Radius",
                   index_to_frag_mapping=None, fp_gen=None, ao=None, # for DB_Specific_Radius only
                   filter_by_MW=None,
                   verbose=True,
                   ):
    """
    Run inference on a given input tensor and visualize the top-k retrieved molecules.
    Hence, shape of inputs is (n, 3) where n is the number of NMR peaks (and other infos)
    """
    
    if verbose:
        print("_________________________________________________________")

    returning_smiles = []
    returning_names = []
    inputs = inputs.unsqueeze(0).to(model.device)
    NMR_type_indicator = NMR_type_indicator.to(model.device)
    rankingset_data = rankingset_data.to(model.device)
    pred = model(inputs, NMR_type_indicator)
    pred = torch.sigmoid(pred) # sigmoid
    pred_FP = torch.where(pred.squeeze()>0.5, 1, 0)
    # print(pred_FP.nonzero().squeeze().tolist())
    if ground_truth_FP is not None:
        print("Prediction's cosine similarity to ground truth: ", compute_cos_sim(ground_truth_FP, pred_FP.to("cpu").float()))
        print("\n\n")
    if filter_by_MW == "from_input":
        mw_from_input = inputs[0][-1][0].item()
        filter_by_MW = [mw_from_input*0.8, mw_from_input*1.2]
    if filter_by_MW is not None and type(filter_by_MW) != list:
        raise ValueError("filter_by_MW must be a list of two elements, or 'from_input' or None!")
    topk = retrieve_top_k_by_rankingset(rankingset_data, pred, smiles_and_names, k=k, filter_by_MW=filter_by_MW)
       
    i=0
    for value, (smile, name, _, _), retrieved_FP in topk:
        if verbose:
            mol = Chem.MolFromSmiles(smile)
    
            if fp_type == "MFP_Specific_Radius":
                img = Draw.MolToImage(mol)
            elif fp_type == "DB_Specific_Radius":
                img = show_retrieved_mol_with_highlighted_frags(smile, retrieved_FP, index_to_frag_mapping, fp_gen, ao)
            else:
                raise ValueError("fp_type must be either MFP_Specific_Radius or DB_Specific_Radius")
        
            print(f"________retival #{i+1}, cosine similarity to prediction: {value.item()}_________________")
            if ground_truth_FP is not None:
                print("________retival's   cosine similarity to ground truth: ", compute_cos_sim(ground_truth_FP, retrieved_FP.to_dense().to("cpu").float()).item())

            print(f"SMILES: {smile}") 
            print(f"Name {name}")
            img.show()
        i+=1
        returning_smiles.append(smile)
        returning_names.append(name)
    return returning_smiles, returning_names
        
### save visualization as PNG 

def visualize_smiles(smiles_list, name_list, file_path):
    """
    Generate a PNG file visualizing a list of SMILES strings.
    
    Args:
        smiles_list (list of str): Ordered list of SMILES strings.
        file_path (str): Path to save the output PNG file.
    """
    # Convert SMILES strings to RDKit molecule objects
    molecules = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    
    # Remove None values if any invalid SMILES strings exist
    molecules = [mol for mol in molecules if mol is not None]
    
    # Generate and save the image
    if molecules:
        legends = [name for name in name_list]
        img = Draw.MolsToGridImage(molecules, molsPerRow=4, subImgSize=(500, 500), legends=legends, returnPNG=False)
        img.save(file_path)
    else:
        raise ValueError("No valid molecules were generated from the input SMILES.")
  
inference_invetigate_path = pathlib.Path(__file__).resolve().parents[0] / "inference_examples"

def save_molecule_inference(smiles, name, index_rkst, model, model_name, inputs, NMR_type_indicator, rankingset_data, smiles_and_names, k=20, mode=None, ground_truth_FP=None, 
                         fp_type="MFP_Specific_Radius", index_to_frag_mapping=None, fp_gen=None, ao=None):
    """
    Investigate a single molecule by running inference on it and visualizing the top-k retrieved molecules.
    
    Args:
        smiles (str): SMILES string of the molecule to investigate.
        name (str): Name of the molecule to investigate.
        index_rkst (int): Index of the molecule in the ranking set.
        model (nn.Module): Model for inference.
        model_name (str): Name of the model for inference.
        inputs (torch.Tensor): Input tensor for inference.
        NMR_type_indicator (torch.Tensor): NMR type indicator tensor for inference.
        rankingset_data (torch.Tensor): Ranking set data for inference.
        smiles_and_names (list of tuples): List of tuples containing SMILES and names of the molecules in the ranking set.
        k (int): Number of top-k retrieved molecules to visualize.
        mode (str): Mode for building input tensor.
        ground_truth_FP (torch.Tensor): Ground truth fingerprint tensor for comparison.
        fp_type (str): Fingerprint type for prediction.
        index_to_frag_mapping (dict): Mapping of fingerprint index to fragments.
        fp_gen (FingerprintGenerator): Fingerprint generator for DB_Specific_Radius.
        ao (AtomEnvironment): Atom environment object for DB_Specific_Radius.
        
    """
    
    # handle path
    curr_mol_path = inference_invetigate_path / name 
    curr_result_path = curr_mol_path / model_name
    os.makedirs(curr_result_path, exist_ok=True)
    
    # write metadata
    if not os.path.exists(curr_mol_path / "meta_data.json"):
        metadata = {
            "smiles": smiles,
            "name": name,
            "index_rkst": index_rkst
        }
        with open(curr_mol_path / "meta_data.json", "w") as f:
            json.dump(metadata, f, indent=4)
        
        if smiles is not None and name not in  ["unknown", "Unknown", "UNKNOWN"]:
            mol = Chem.MolFromSmiles(smiles)
            img = Draw.MolToImage(mol)
            img.save(curr_mol_path / "molecule_ground_truth.png")
    
    # run inference 
    returning_smiles, returning_names = inference_topK(inputs, NMR_type_indicator, model, rankingset_data, smiles_and_names, 
                                      k=k, mode=mode, ground_truth_FP=ground_truth_FP,
                                      fp_type=fp_type, index_to_frag_mapping=index_to_frag_mapping, 
                                      fp_gen=fp_gen, ao=ao, verbose=False, filter_by_MW="from_input")
    visualize_smiles(returning_smiles, returning_names, curr_result_path / "visualization.png")
    
    return returning_smiles


# generate temp_hsqc.txt for deepsat website
def convert_hsqc_tensort_to_txt(split, index):
    hsqc_path = f"/workspace/SMILES_dataset/{split}/HSQC/{index}.pt"
    hsqc = torch.load(hsqc_path)
    # write to txt
    with open(f"tmp_hsqc_{index}.txt", "w") as f:
        f.write("13C,1H,Intensity\n")
        f.write("\n".join([str(i)[1:-1] for i in hsqc.tolist()]))
        
    with open(f"tmp_hsqc_{index}_separate.txt", "w") as f:
        f.write("13C\n")
        f.write(str(hsqc[:,0].tolist()))
        f.write("\n")
        f.write("1H\n")
        f.write(str(hsqc[:,1].tolist()))
        f.write("\n")
        f.write("Intensity\n")
        f.write(str(hsqc[:,2].tolist()))
        
# convert_hsqc_tensort_to_txt("/workspace/SMILES_dataset/test/HSQC/10018.pt")
    