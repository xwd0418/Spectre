import torch, os, heapq
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import Draw
import numpy as np
import sys, pathlib, json, yaml
from pathlib import Path

### model selection ###

def find_checkpoint_path_DB_specific_FP(model_type):
    match model_type:
        case "only_1d":
            checkpoint_path = Path("/root/gurusmart/MorganFP_prediction/reproduce_previous_works/kekulize_smiles/train_on_all_data_possible/only_1d_trial_1/checkpoints/epoch=81-step=59860.ckpt")
        case "HSQC":
            checkpoint_path = Path("/root/gurusmart/MorganFP_prediction/reproduce_previous_works/kekulize_smiles/train_on_all_data_possible/only_hsqc_trial_1/checkpoints/epoch=80-step=69417.ckpt")
        case "C-NMR":
            checkpoint_path = Path("/root/gurusmart/MorganFP_prediction/reproduce_previous_works/kekulize_smiles/train_on_all_data_possible/only_c_trial_1/checkpoints/epoch=68-step=55752.ckpt")
        case _:
            raise ValueError(f"model_type: {model_type} not recognized")
        
    return checkpoint_path

def find_checkpoint_path_entropy_based_FP(model_type):
    match model_type:
        case "C-NMR":
            checkpoint_path = Path("/root/gurusmart/MorganFP_prediction/reproduce_previous_works/stop_on_cosine/all_data_possible/only_c_trial_1/checkpoints/epoch=97-step=79184.ckpt")
        case "H-NMR":
            checkpoint_path = Path("/root/gurusmart/MorganFP_prediction/reproduce_previous_works/stop_on_cosine/all_data_possible/only_h_trial_1/checkpoints/epoch=86-step=63510.ckpt")
        case "HSQC":
            checkpoint_path = Path("/root/gurusmart/MorganFP_prediction/reproduce_previous_works/stop_on_cosine/all_data_possible/only_hsqc_trial_1/checkpoints/epoch=62-step=53991.ckpt")
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
from datasets.hsqc_folder_dataset import FolderDataModule

from datasets.dataset_utils import fp_loader_configer



def choose_model_DB_specific_FP(model_type):
    checkpoint_path = find_checkpoint_path_DB_specific_FP(model_type)
    model_path = checkpoint_path.parents[1]
    hyperpaerameters_path = model_path / "hparams.yaml"
    
    with open(hyperpaerameters_path, 'r') as file:
        hparams = yaml.safe_load(file)
        
    del hparams['checkpoint_path'] # prevent double defition of checkpoint_path
    hparams['use_peak_values'] = False
    hparams['num_workers'] = 0
    model = OptionalInputRankedTransformer.load_from_checkpoint(checkpoint_path, **hparams)
    
    fp_loader = fp_loader_configer.fp_loader
    max_radius = int(hparams['FP_choice'].split("_")[-1])
    fp_loader.setup(hparams['out_dim'], max_radius)
    
        
    model.eval()
    return hparams, model
        
            
def choose_model_entropy_based_FP(model_type, include_test_loader=True, shuffle_loader=False):
    
    checkpoint_path = find_checkpoint_path_entropy_based_FP(model_type)
    
    model_path = checkpoint_path.parents[1]
    hyperpaerameters_path = model_path / "hparams.yaml"

    # checkpoint_path = model_path / "checkpoints/epoch=14-step=43515.ckpt"


    with open(hyperpaerameters_path, 'r') as file:
        hparams = yaml.safe_load(file)
        
    FP_building_type = hparams['FP_building_type'].split("_")[-1]
    only_2d = False
    # only_2d = True
    print(FP_building_type)
    max_radius = int(hparams['FP_choice'].split("_")[-1][1:])
    print("max_radius: ", max_radius)
    
    specific_radius_mfp_loader = fp_loader_configer.fp_loader
    if  max_radius!=specific_radius_mfp_loader.max_radius or only_2d!=specific_radius_mfp_loader.only_2d:
        specific_radius_mfp_loader.setup(only_2d=only_2d,FP_building_type=FP_building_type)
        specific_radius_mfp_loader.set_max_radius(int(hparams['FP_choice'].split("_")[-1][1:]), only_2d=only_2d)


    del hparams['checkpoint_path'] # prevent double defition of checkpoint_path
    hparams['use_peak_values'] = False
    hparams['num_workers'] = 0
    if "test_on_deepsat_retrieval_set" not in hparams:
        hparams['test_on_deepsat_retrieval_set'] = False
    if "rank_by_test_set" not in hparams:
        hparams['rank_by_test_set'] = False
    model = OptionalInputRankedTransformer.load_from_checkpoint(checkpoint_path, **hparams)
    
    if not include_test_loader:
        return hparams, model
    
    if model_type == "HSQC":
        # datamodule = FolderDataModule(dir="/workspace/SMILES_dataset", FP_choice=hparams["FP_choice"], input_src=["HSQC"], batch_size=hparams['bs'], parser_args=hparams, persistent_workers=False)

        # # datamodule = OptionalInputDataModule(dir="/workspace/SMILES_dataset", FP_choice=hparams["FP_choice"], input_src=["HSQC", "oneD_NMR"], batch_size=hparams['bs'], parser_args=hparams)
        # # datamodule.setup("test")
        # # loader_all_inputs, loader_HSQC_H_NMR, loader_HSQC_C_NMR, loader_only_hsqc, loader_only_1d, loader_only_H_NMR, loader_only_C_NMR = datamodule.test_dataloader()
        
        # datamodule.setup("test")
        # test_loader = datamodule.test_dataloader()
        input_src=["HSQC"]
    else:
        input_src=["HSQC", "oneD_NMR"]
    datamodule = OptionalInputDataModule(dir="/workspace/SMILES_dataset", FP_choice=hparams["FP_choice"], input_src=["HSQC", "oneD_NMR"], batch_size=1, parser_args=hparams)
    datamodule.setup("predict")
    loader_all_inputs, loader_HSQC_H_NMR, loader_HSQC_C_NMR, loader_only_hsqc, loader_only_1d, loader_only_H_NMR, loader_only_C_NMR = \
        datamodule.predict_dataloader(shuffle=shuffle_loader)
        
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

def retrieve_top_k_by_rankingset(data, prediction_query, smiles_and_names, k=30, filter_by_MW=None, weight_pred = None):
    # data means the rankingset data
    # Expect filter by MW to be [lower_bnound, upper_bound]
    query = F.normalize(prediction_query, dim=1, p=2.0).squeeze()
    # print("query shape: ", query.shape, query.dtype)
    if weight_pred is not None:
        query = query * weight_pred
        # print("query shape after weight: ", query.shape, query.dtype)
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
    fp1 = fp1.float()
    fp2 = fp2.float()
    return (fp1 @ fp2) / (torch.norm(fp1) * torch.norm(fp2)).item()


from notebook_and_scripts.SMILES_fragmenting.build_dataset_specific_FP.find_frags import get_fragments_for_each_atom_id, get_fragments_for_each_atom_id_v2
import io
from PIL import Image
from math import sqrt
from rdkit.Chem.Draw import SimilarityMaps

def show_retrieved_mol_with_highlighted_frags(predicted_FP, retrieval_smiles, show_H=False):
    '''
    used in db-specific FP
    This functions visualizes the retrieved molecule with the predicted fragments highlighted
    '''
    def show_png(data):
        
        bio = io.BytesIO(data)
        img = Image.open(bio)
        return img

    def set_based_cosine(x,y):
        '''x, y are same shape array'''
        a = set(x)
        b = set(y)
        return (len(a&b))/(sqrt(len(a))*sqrt(len(b)))
    
    fp_loader = fp_loader_configer.fp_loader
    
    predicted_frag_indices = set(predicted_FP.nonzero()[:,0].tolist())
    retrieval_FP = fp_loader.build_mfp_for_new_SMILES(retrieval_smiles)
    
    # Step 1: Create molecule and hydrogenated copy
    retrieval_mol_h = Chem.MolFromSmiles(retrieval_smiles)
    retrieval_mol_h = Chem.AddHs(retrieval_mol_h)  # Keep explicit Hs for fragment mapping
    
    weights_h = [0] * retrieval_mol_h.GetNumAtoms()
    base_sim = set_based_cosine(predicted_frag_indices, retrieval_FP.nonzero()[:,0].tolist())
    # Step 2: Compute weights on the molecule WITH hydrogens
    atom_to_frags, all_frags  = get_fragments_for_each_atom_id_v2(retrieval_smiles)
    for atom_id, frags in atom_to_frags.items():
        frag_indices_with_this_atom = {fp_loader.frag_to_index_map[frag] for frag in (all_frags-frags) if frag in fp_loader.frag_to_index_map }
        sim_without_this_atom = set_based_cosine(frag_indices_with_this_atom, predicted_frag_indices)
        
        # sim_without_this_atom = cosine((all_frags-frags).intersection(fp_loader_frags), {fp_loader.index_to_frag_mapping[i] for i in predicted_frag_indices})
        weights_h[atom_id] = base_sim - sim_without_this_atom

    weights_h, max_weight = SimilarityMaps.GetStandardizedWeights(weights_h)
    
    if not show_H:
        # Step 3: Remove explicit hydrogens and map weights
        retrieval_mol = Chem.RemoveHs(retrieval_mol_h)
        
        # Step 4: Map hydrogenated weights to non-hydrogenated molecule
        heavy_atom_map = retrieval_mol_h.GetSubstructMatch(retrieval_mol)  # Maps H-heavy to no-H
        
        weights = [weights_h[i] for i in heavy_atom_map]  # Remap weights to no-H molecule

    # Step 5: Draw similarity map on molecule without hydrogens
    d = Draw.MolDraw2DCairo(400, 400)
    SimilarityMaps.GetSimilarityMapFromWeights(retrieval_mol, weights, draw2d=d)
    
    d.FinishDrawing()
    img = (show_png(d.GetDrawingText()))
    
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
                   filter_by_MW=None,
                   verbose=True,
                   weight_pred = None
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
    topk = retrieve_top_k_by_rankingset(rankingset_data, pred, smiles_and_names, k=k, filter_by_MW=filter_by_MW, weight_pred = weight_pred)
       
    i=0
    for value, (smile, name, _, _), retrieved_FP in topk:
        if verbose:
            mol = Chem.MolFromSmiles(smile)
    
            if fp_type == "MFP_Specific_Radius":
                img = Draw.MolToImage(mol)
            elif fp_type == "DB_Specific_Radius":
                img = show_retrieved_mol_with_highlighted_frags(retrieved_FP, smile)
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
                         fp_type="MFP_Specific_Radius", ):
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
                                      fp_type=fp_type,  
                                      verbose=False, filter_by_MW="from_input")
    visualize_smiles(returning_smiles, returning_names, curr_result_path / "visualization.png")
    
    return returning_smiles


# generate temp_hsqc.txt for deepsat website
def convert_hsqc_tensort_to_txt(path, name_info):
    hsqc_path = path.replace("oneD_NMR", "HSQC")
    hsqc = torch.load(hsqc_path)
    # write to txt
    with open(f"HSQC_for_deepsat/hsqc_{name_info}.txt", "w") as f:
        f.write("13C,1H,Intensity\n")
        f.write("\n".join([str(i)[1:-1] for i in hsqc.tolist()]))
        
    # with open(f"tmp_hsqc_{index}_separate.txt", "w") as f:
    #     f.write("13C\n")
    #     f.write(str(hsqc[:,0].tolist()))
    #     f.write("\n")
    #     f.write("1H\n")
    #     f.write(str(hsqc[:,1].tolist()))
    #     f.write("\n")
    #     f.write("Intensity\n")
    #     f.write(str(hsqc[:,2].tolist()))
        
# convert_hsqc_tensort_to_txt("/workspace/SMILES_dataset/test/HSQC/10018.pt")
    