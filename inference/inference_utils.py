import torch, os, heapq
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import Draw
import numpy as np
import sys, pathlib, json, yaml
from pathlib import Path
from io import BytesIO
from PIL import Image
import base64


### model selection ###
def find_checkpoint_path_entropy_on_hashes_FP(model_type):
    match model_type:
        case "backend":
            checkpoint_path = Path("/home/ad.ucsd.edu/w6xu/model_weights/flexible_model_flexible_MW_entropy_on_hash/checkpoints/spectre_model_weights.ckpt")
        case "optional":
            # checkpoint_path = Path("/root/gurusmart/MorganFP_prediction/reproduce_previous_works/entropy_on_hashes/flexible_models_jittering_size_1/r0_r6_trial_1/checkpoints/epoch=95-step=21696.ckpt")
            # checkpoint_path = Path("/root/gurusmart/MorganFP_prediction/reproduce_previous_works/entropy_on_hashes/flexible_models_jittering_flexible_MW/r0_r6_trial_1/checkpoints/epoch=73-step=16724.ckpt")
            checkpoint_path = Path("/root/gurusmart/MorganFP_prediction/reproduce_previous_works/entropy_on_hashes/flexible_models_jittering_flexible_MW_flexible_normal_hsqc/r0_r6_trial_1/checkpoints/epoch=95-step=21696.ckpt")
        case "C-NMR":
            # checkpoint_path = Path("/root/gurusmart/MorganFP_prediction/reproduce_previous_works/entropy_on_hashes/train_on_all_data_possible/only_c_trial_2/checkpoints/epoch=79-step=64640.ckpt")
            checkpoint_path = Path("/root/gurusmart/MorganFP_prediction/reproduce_previous_works/entropy_on_hashes/train_on_all_data_possible_with_jittering/only_c_trial_1/checkpoints/epoch=90-step=73528.ckpt")
        case "HSQC":
            checkpoint_path = Path("/root/gurusmart/MorganFP_prediction/reproduce_previous_works/entropy_on_hashes/all_HSQC_jittering_search/only_hsqc_jittering_1-trial-1/checkpoints/epoch=82-step=35607.ckpt")
        case _:
            raise ValueError(f"model_type: {model_type} not recognized")
        
    return checkpoint_path


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



def choose_model(model_type, fp_type="entropy_on_hashes", return_data_loader=False, 
                 use_predict_loader=True,
                 should_shuffle_loader=False, 
                 checkpoint_path=None, 
                 load_for_moonshot=False, load_untrained_model=False, ):
    
    if checkpoint_path is None:
        if fp_type == "DB_specific_FP":
            checkpoint_path = find_checkpoint_path_DB_specific_FP(model_type)
        if fp_type == "entropy_on_hashes":
            checkpoint_path = find_checkpoint_path_entropy_on_hashes_FP(model_type)
    else:
        checkpoint_path = Path(checkpoint_path)
    model_path = checkpoint_path.parents[1]
    hyperpaerameters_path = model_path / "hparams.yaml"
    
    with open(hyperpaerameters_path, 'r') as file:
        hparams = yaml.safe_load(file)
        
    del hparams['checkpoint_path'] # prevent double defition of checkpoint_path
    # hparams['use_peak_values'] = False
    print("loading model from: ", checkpoint_path)
    
    if load_untrained_model:
        model = OptionalInputRankedTransformer(**hparams,  fp_loader = None, save_params=False)
        return model
    if load_for_moonshot:
        model = OptionalInputRankedTransformer.load_from_checkpoint(checkpoint_path, fp_loader = None,  **hparams, save_params=False)
        return model
    
    fp_loader = fp_loader_configer.fp_loader
    hparams['num_workers'] = 0
    model = OptionalInputRankedTransformer.load_from_checkpoint(checkpoint_path, fp_loader = fp_loader,  **hparams)
    max_radius = int(hparams['FP_choice'].split("_")[-1])
    fp_loader.setup(hparams['out_dim'], max_radius)
    
        
    model.eval()
    if not return_data_loader:
        return hparams, model
    
    test_loader = get_data_loader(model_type, should_shuffle_loader, hparams, use_predict_loader)
    return hparams, model, test_loader
        
            

def get_data_loader(model_type, should_shuffle_loader, hparams, use_predict_loader):
    # if model_type == "HSQC":
    #     # datamodule = FolderDataModule(dir="/workspace/SMILES_dataset", FP_choice=hparams["FP_choice"], input_src=["HSQC"], batch_size=hparams['bs'], parser_args=hparams, persistent_workers=False)
    #     # # datamodule = OptionalInputDataModule(dir="/workspace/SMILES_dataset", FP_choice=hparams["FP_choice"], input_src=["HSQC", "oneD_NMR"], batch_size=hparams['bs'], parser_args=hparams)
    #     # # datamodule.setup("test")
    #     # # loader_all_inputs, loader_HSQC_H_NMR, loader_HSQC_C_NMR, loader_only_hsqc, loader_only_1d, loader_only_H_NMR, loader_only_C_NMR = datamodule.test_dataloader()
    #     # datamodule.setup("test")
    #     # test_loader = datamodule.test_dataloader()
    #     input_src=["HSQC"]
    # else:
    #     input_src=["HSQC", "oneD_NMR"]
    
    datamodule = OptionalInputDataModule(dir="/workspace/SMILES_dataset", FP_choice=hparams["FP_choice"], input_src=["HSQC", "oneD_NMR"], fp_loader = fp_loader_configer.fp_loader, batch_size=1, parser_args=hparams)
    if use_predict_loader:
        datamodule.setup("predict")
        loader_all_inputs, loader_HSQC_H_NMR, loader_HSQC_C_NMR, loader_only_hsqc, loader_only_1d, loader_only_H_NMR, loader_only_C_NMR = \
            datamodule.predict_dataloader(shuffle=should_shuffle_loader)
    else:
        datamodule.setup("test")
        loader_all_inputs, loader_HSQC_H_NMR, loader_HSQC_C_NMR, loader_only_hsqc, loader_only_1d, loader_only_H_NMR, loader_only_C_NMR = \
            datamodule.test_dataloader()
            
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
        
        case "optional":
            test_loader = loader_all_inputs
                        
        case _:
            raise ValueError(f"model_type: {model_type} not recognized")
    return test_loader




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
    
    if 0 in indices_location:
        hsqc_start, hsqc_end = indices_location[0]
        hsqc = inputs[hsqc_start:hsqc_end]
    else:
        hsqc = None
    if 1 in indices_location:
        c_nmr_start, c_nmr_end = indices_location[1]
        c_tensor = inputs[c_nmr_start:c_nmr_end,0]
    else:
        c_tensor = None
    if 2 in indices_location:
        h_nmr_start, h_nmr_end = indices_location[2]
        h_tensor = inputs[h_nmr_start:h_nmr_end,1]
    else:
        h_tensor = None

    
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

def retrieve_top_k_by_rankingset(data, prediction_query, smiles_and_names, k=30, filter_by_MW=None, weighting_pred = None):
    # data means the rankingset data
    # Expect filter by MW to be [lower_bnound, upper_bound]
    query = F.normalize(prediction_query, dim=1, p=2.0).squeeze()
    # print("query shape: ", query.shape, query.dtype)
    if weighting_pred is not None:
        query = query * weighting_pred
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
    
    return ((fp1 @ fp2).item() / (torch.norm(fp1) * torch.norm(fp2))).item()


from notebook_and_scripts.SMILES_fragmenting.build_dataset_specific_FP.find_frags import  get_bitInfos_for_each_atom_idx
import io
from PIL import Image
from math import sqrt
from rdkit.Chem.Draw import SimilarityMaps

def show_retrieved_mol_with_highlighted_frags(predicted_FP, retrieval_smiles, need_to_clean_H=False, img_size=400): 
    '''
    used in db-specific FP
    This functions visualizes the retrieved molecule with the predicted fragments highlighted
    '''
    def show_png(data):
        
        bio = io.BytesIO(data)
        img = Image.open(bio)
        return img
    fp_loader = fp_loader_configer.fp_loader
    
    # retrieval_FP = fp_loader.build_mfp_for_new_SMILES(retrieval_smiles)
    
    # Step 1: Create molecule and hydrogenated copy
    retrieval_mol = Chem.MolFromSmiles(retrieval_smiles)
    
    atom_to_bit_infos,_ = get_bitInfos_for_each_atom_idx(retrieval_smiles)
    retrieval_FP = fp_loader.build_mfp_from_bitInfo(atom_to_bit_infos)
    baseSimilarity = compute_cos_sim(retrieval_FP, predicted_FP.cpu())
    # print("base similarity: ", baseSimilarity)
    weights = [
        baseSimilarity - compute_cos_sim(predicted_FP.cpu(), fp_loader.build_mfp_from_bitInfo(atom_to_bit_infos, [atomId]))
        for atomId in range(retrieval_mol.GetNumAtoms())
    ] 
    

    weights, max_weight = SimilarityMaps.GetStandardizedWeights(weights)
    # Step 5: Draw similarity map on molecule without hydrogens
    d = Draw.MolDraw2DCairo(img_size, img_size)
    # SimilarityMaps.GetSimilarityMapFromWeights(retrieval_mol, weights, draw2d=d, contourLines=0)
    draw_high_res_similarity_map(retrieval_mol, weights, draw2d=d, contourLines=0,)
    
    d.FinishDrawing()
    img = (show_png(d.GetDrawingText()))
    
    # img.show()
    return img
         
         
from rdkit import Chem, Geometry
from rdkit.Chem import Draw, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from matplotlib import cm
import matplotlib

def draw_high_res_similarity_map(mol, weights, draw2d, colorMap=None, 
                                sigma=None, contourLines=10, gridResolution = 0.05
                                ):
  """
  copied from /opt/conda/lib/python3.11/site-packages/rdkit/Chem/Draw/SimilarityMaps.py  => GetSimilarityMapFromWeights
  """
  if mol.GetNumAtoms() < 2:
    raise ValueError("too few atoms")

  if draw2d is None:
    raise ValueError("the draw2d argument must be provided")
  mol = rdMolDraw2D.PrepareMolForDrawing(mol, addChiralHs=False)
  if not mol.GetNumConformers():
    rdDepictor.Compute2DCoords(mol)
  if sigma is None:
    if mol.GetNumBonds() > 0:
      bond = mol.GetBondWithIdx(0)
      idx1 = bond.GetBeginAtomIdx()
      idx2 = bond.GetEndAtomIdx()
      sigma = 0.3 * (mol.GetConformer().GetAtomPosition(idx1) -
                     mol.GetConformer().GetAtomPosition(idx2)).Length()
    else:
      sigma = 0.3 * (mol.GetConformer().GetAtomPosition(0) -
                     mol.GetConformer().GetAtomPosition(1)).Length()
    sigma = round(sigma, 2)

  sigmas = [sigma] * mol.GetNumAtoms()
  locs = []
  for i in range(mol.GetNumAtoms()):
    p = mol.GetConformer().GetAtomPosition(i)
    locs.append(Geometry.Point2D(p.x, p.y))
  draw2d.ClearDrawing()
  ps = Draw.ContourParams()
  ps.fillGrid = True
  ps.gridResolution = gridResolution
  ps.extraGridPadding = 0.5

  if colorMap is not None:
    if cm is not None and isinstance(colorMap, type(cm.Blues)):
      # it's a matplotlib colormap:
      clrs = [tuple(x) for x in colorMap([0, 0.5, 1])]
    elif type(colorMap) == str:
      if cm is None:
        raise ValueError("cannot provide named colormaps unless matplotlib is installed")
      clrs = [tuple(x) for x in matplotlib.colormaps[colorMap]([0, 0.5, 1])]
    else:
      clrs = [colorMap[0], colorMap[1], colorMap[2]]
    ps.setColourMap(clrs)

  Draw.ContourAndDrawGaussians(draw2d, locs, weights, sigmas, nContours=contourLines, params=ps)
  draw2d.drawOptions().clearBackground = False
  draw2d.DrawMolecule(mol)
  return draw2d           
    
######  for unknow compound ######    
def build_input(compound_dir, mode = None, include_hsqc = True, include_c_nmr = True, include_h_nmr = True, include_MW = True):
    """
    build input tensor for unkonw compound
    mode is either None, "no_sign", "flip_sign"
    
    return inputs, NMR_type_indicator
    """
    # print("\n\n")
    # print(compound_dir.split("/")[-1])
    # print("\n")
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
    with open(os.path.join(compound_dir, "mw.txt"), 'r') as file:
            # Read the content of the file
            content = file.read()
            # Convert the content to a float
            mw = float(content)
            
    NMR_type_indicator, inputs = get_inputs_and_indicators_from_NMR_tensors(include_hsqc, include_c_nmr, include_h_nmr, include_MW, hsqc, c_tensor, h_tensor, mw)
    
    # print(inputs)
    # print(hsqc, c_tensor, h_tensor)
    # plot_NMR(hsqc, c_tensor[:,0], h_tensor[:,0])
    return inputs, NMR_type_indicator

def get_inputs_and_indicators_from_NMR_tensors(include_hsqc, include_c_nmr, include_h_nmr, include_MW, hsqc, c_tensor, h_tensor, mw):
    input_NMRs = []
    NMR_type_indicator = []
    if include_hsqc:
        input_NMRs.append(hsqc)
        # hsqc_type = 4 if (hsqc[:2]==0).all() else 0 # if multiplicity is all 0s, it is normal hsqc
        hsqc_type = 0
        NMR_type_indicator+= [hsqc_type]*hsqc.shape[0]
        
    if include_c_nmr:
        input_NMRs.append(c_tensor)
        NMR_type_indicator+= [1]*c_tensor.shape[0]
    if include_h_nmr:
        input_NMRs.append(h_tensor)
        NMR_type_indicator+= [2]*h_tensor.shape[0]
    inputs = torch.vstack(input_NMRs)   
    if include_MW:
        mol_weight = torch.tensor([mw,0,0]).float()
        inputs = torch.vstack([inputs, mol_weight])
        input_NMRs.append(mol_weight)
        NMR_type_indicator.append(3)
    NMR_type_indicator = torch.tensor(NMR_type_indicator)
    return NMR_type_indicator,inputs

def predict_FP(inputs, NMR_type_indicator, model):
    inputs = inputs.unsqueeze(0).to(model.device)
    NMR_type_indicator = NMR_type_indicator.to(model.device)
    pred = model(inputs, NMR_type_indicator)
    pred = torch.sigmoid(pred) # sigmoid
    return pred

def inference_topK(inputs, NMR_type_indicator, model, rankingset_data, smiles_and_names, 
                   k=5, mode = None, ground_truth_FP=None,
                   similarity_mapping_showing = "MFP_Specific_Radius",
                   filter_by_MW=None,
                   verbose=True,
                   weighting_pred = None,
                   infer_in_backend_service = False,
                   encode_img = True,
                   img_size = 800,
                   ):
    """
    Run inference on a given input tensor and visualize the top-k retrieved molecules.
    Hence, shape of inputs is (n, 3) where n is the number of NMR peaks (and other infos)
    """
    
    if verbose:
        print("_________________________________________________________")

    
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
    topk = retrieve_top_k_by_rankingset(rankingset_data, pred, smiles_and_names, k=k, filter_by_MW=filter_by_MW, weighting_pred = weighting_pred)
       
    return return_infos_from_topk(topk, pred, ground_truth_FP, similarity_mapping_showing, verbose, encode_img, img_size, infer_in_backend_service)

def return_infos_from_topk(topk, pred, ground_truth_FP = None, similarity_mapping_showing = "both", verbose = False, encode_img = True, img_size = 800, infer_in_backend_service = True):
    
    returning_smiles = []
    returning_names = []
    returning_imgs = []
    returning_values = []
    returning_MWs = []
    
    i=0
    for value, (smile, name, mw, _), retrieved_FP in topk:
        if verbose or infer_in_backend_service:
            mol = Chem.MolFromSmiles(smile)
    
            if similarity_mapping_showing in ["MFP_Specific_Radius", 0, False]:
                img = Draw.MolToImage(mol)
            elif similarity_mapping_showing in ["DB_Specific_Radius", 1, True]:
                img = show_retrieved_mol_with_highlighted_frags(pred[0], smile, img_size=img_size) #
            elif similarity_mapping_showing == "both":
                img_no_sim_map = Draw.MolToImage(mol, size=(img_size,img_size))
                img_with_sim_map = show_retrieved_mol_with_highlighted_frags(pred[0], smile, img_size=img_size) #
                
            else:
                raise ValueError("fp_type must be either MFP_Specific_Radius or DB_Specific_Radius")
        

            if verbose:
                print(f"________retival #{i+1}, cosine similarity to prediction: {value.item()}_________________")
                if ground_truth_FP is not None:
                    print("________retival's   cosine similarity to ground truth: ", compute_cos_sim(ground_truth_FP, retrieved_FP.to_dense().to("cpu").float()))

                print(f"SMILES: {smile}") 
                print(f"Name {name}")
                img.show()
            if infer_in_backend_service:
                returning_MWs.append(mw)
                if encode_img:
                    
                    if similarity_mapping_showing == "both":
                        img_no_sim_map_base64 = encode_image(img_no_sim_map)
                        img_with_sim_map_base64 = encode_image(img_with_sim_map)
                        returning_imgs.append([img_no_sim_map_base64, img_with_sim_map_base64])
                    else:
                        img_base64 = encode_image(img)
                        returning_imgs.append(img_base64)
                        
                else:
                    returning_imgs.append(img)
                returning_values.append(value.item())
        i+=1
        returning_smiles.append(smile)
        returning_names.append(name)
    if infer_in_backend_service:
        return returning_smiles, returning_names, returning_imgs, returning_MWs, returning_values
    return returning_smiles, returning_names

def encode_image(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    return img_base64

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
                                      similarity_mapping_showing=fp_type,  
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
    