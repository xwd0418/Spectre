'''Backend for SPECTRE, a web application for using NMR spectra to predict molecular structure.'''

# chemistry
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Draw
import sys, pickle, os
#backend framwork
from flask import Flask, request, jsonify
from flask_cors import CORS

import torch, numpy as np
import yaml
import base64
from io import BytesIO
from PIL import Image
torch.set_printoptions(precision=10)
torch.set_float32_matmul_precision('medium')
    

# Add the directory containing the module to sys.path
import sys, pathlib
root_path = pathlib.Path(__file__).resolve().parents[2]
repo_path = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(repo_path))
from datasets.dataset_utils import specific_radius_mfp_loader
from models.optional_input_ranked_transformer import OptionalInputRankedTransformer
from datasets.optional_2d_folder_dataset import OptionalInputDataModule
from datasets.hsqc_folder_dataset import FolderDataModule
from pytorch_lightning.loggers import TensorBoardLogger

# helper functions 
from data_process import retrieve_top_k_by_rankingset, build_input, plot_NMR, convert_to_tensor_1d_nmr
from flask_utils import build_cors_preflight_response, build_actual_response

app = Flask(__name__)
CORS(app)

from flask import Response

@app.before_request
def basic_authentication():
    if request.method.lower() == 'options':
        return Response()
    
'''for a single model, show top-5'''
def show_topK(inputs, k=5, mode = "no_sign"):
    print("_________________________________________________________")

    
    inputs = inputs.unsqueeze(0)#.to("cuda")
    pred = model(inputs)
    pred=torch.sigmoid(pred) # sigmoid
    # pred_FP = torch.where(pred.squeeze()>0.5, 1, 0)
    # print(pred_FP.nonzero().squeeze().tolist())
    
    topk = retrieve_top_k_by_rankingset(rankingset_data, pred, smiles_and_names, k=k)
    
    i=0
    all_retrivals = []
    for ite, (value, (smile, name, _, _), predicted_FP) in enumerate(topk):
        # print(f"____________________________retival #{i+1}, cosine similarity: {value.item()}_____________________________")
        mol = Chem.MolFromSmiles(smile)
        # print("retrived FP", predicted_FP.squeeze().tolist())
        # print(f"SMILES: {smile}")
        # print(f"Name {name}")
        hsqc_path, oned_path = smiles_to_NMR_path[smile]
        #check is path file exists
        hsqc, c_tensor, h_tensor = None, None, None
        # print("retrieved path: ",oned_path)
        if os.path.exists(hsqc_path):
            hsqc = torch.load(hsqc_path)
        if os.path.exists(oned_path):
            c_tensor, h_tensor = torch.load(oned_path)
            
        if hsqc is not None:
            if mode == "no_sign":
                hsqc = torch.abs(hsqc)
            elif mode == "flip_sign":
                hsqc[:,2] = -hsqc[:,2]
        img = Draw.MolToImage(mol)
        
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')

        i+=1
        
        all_retrivals.append({"smile": smile, "name": name, "image": img_base64})
    return all_retrivals

@app.route('/api/hello', methods=['GET','OPTIONS'])
def hello():
    return build_actual_response(jsonify({'retrievals': "hello"}))

@app.route('/api/generate-retrievals', methods=['POST','OPTIONS'])
def generate_image():
    print("bakend received")
    # dummy return 
    buffered = BytesIO()
    img = Image.new('RGB', (256, 256), color='red')
    img.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    retrievals = [{"smile": "what is my smiles", "name": "what is my name", "image": img_base64}]
    res = jsonify({'retrievals': retrievals, "NMR_plt": img_base64})
    return build_actual_response(res)
    
    # print("request.method ",request.method )
    # if request.method == 'OPTIONS':
    #     return build_cors_preflight_response()
    data = request.get_json()
    HSQC = data['HSQC']
    C_NMR = data['C_NMR']
    H_NMR = data['H_NMR']
    mw = data['MW']
    HSQC_format = data['HSQC_format']
    # print("HSQC_format", HSQC_format)
    
    # "option1" -> 1H, 13C, peak
    # "option2" -> 13C, 1H, peak
    # "option3" -> 1H, 13C
    # "option4" -> 13C, 1H
    
    # my model format is 13C, 1H, peak
    swap_c_h = HSQC_format in ["option1", "option3"]
    no_peak_signs = HSQC_format in ["option3", "option4"]
    
    # convert input strings to tensor
    if HSQC == "":
        hsqc = None
    else:
        hsqc_stacked = []
        for peak in HSQC.split("\n"):
            if peak.strip() == "":
                continue
            hsqc_stacked.append(np.fromstring(peak, sep=", "))
        # print(hsqc_stacked)
        hsqc = torch.tensor(np.array(hsqc_stacked)).float()  
        if swap_c_h:         
            hsqc[:,[0,1]] = hsqc[:,[1,0]]
        if no_peak_signs:
            hsqc = torch.cat([hsqc, torch.ones(len(hsqc)).unsqueeze(1)], dim=1)
            
    c_tensor = None if C_NMR=="" else convert_to_tensor_1d_nmr(C_NMR) 
    h_tensor = None if H_NMR=="" else convert_to_tensor_1d_nmr(H_NMR)
    
    inputs = build_input(hsqc, c_tensor, h_tensor, float(mw))
    retrieved_molecules = show_topK(inputs, k=5, mode = "no_sign")
    nmr_fig_str= plot_NMR(hsqc, c_tensor, h_tensor)
    res = jsonify({'retrievals': retrieved_molecules, "NMR_plt": nmr_fig_str})
    return build_actual_response(res)

if __name__ == '__main__':
    from waitress import serve
    # # step 1: load model and datamodule   (here we assume we use all three NMRs)
    # model_path = Path(f"/{root_path}/exps/weird_H_and_tautomer_cleaned/flexible_models_best_FP/r0_r2_FP_trial_2/")
    # hyperpaerameters_path = model_path / "hparams.yaml"
    # # checkpoint_path = model_path / "checkpoints/epoch=14-step=43515.ckpt"

    # with open(hyperpaerameters_path, 'r') as file:
    #     hparams = yaml.safe_load(file)
        
    # FP_building_type = hparams['FP_building_type'].split("_")[-1]
    # only_2d = not hparams['use_oneD_NMR_no_solvent']
    # print("FP_building_type", FP_building_type)
    # print("FP_choice: 0~",int(hparams['FP_choice'].split("_")[-1][1:]))
    # print('setting up specific_radius_mfp_loader...')
    # specific_radius_mfp_loader.setup(only_2d=only_2d,FP_building_type=FP_building_type)
    # print('set up specific_radius_mfp_loader')
    # specific_radius_mfp_loader.set_max_radius(int(hparams['FP_choice'].split("_")[-1][1:]), only_2d=only_2d)

    # del hparams['checkpoint_path'] # prevent double defition of checkpoint_path
    # hparams['use_peak_values'] = False
    # checkpoint_path = model_path / "checkpoints/epoch=42-all_inputs.ckpt"
    # model = OptionalInputRankedTransformer.load_from_checkpoint(checkpoint_path, **hparams)
    # model.to("cpu")

    # print("model device: ", model.device)

    # # step 2: load rankingset
    # chemical_names_lookup = pickle.load(open(f'/workspace/SMILES_dataset/test/Chemical/index.pkl', 'rb'))
    # smiles_and_names = pickle.load(open(f'{root_path}/inference_data/SMILES_chemical_names_remove_stereoChemistry.pkl', 'rb'))
    # rankingset_path = f'{root_path}/inference_data/max_radius_2_only_2d_False_together_no_stereoChemistry_dataset/FP.pt'
    # rankingset_data = torch.load(rankingset_path)#.to("cuda")
    # smiles_to_NMR_path = pickle.load(open(f'{root_path}/inference_data/SMILES_chemical_to_NMR_paths.pkl','rb'))

    print("starting server")
    # serve(app, host="0.0.0.0", port=6660)
    app.run(host='0.0.0.0', port=6660)
