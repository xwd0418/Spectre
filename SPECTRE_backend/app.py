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

torch.set_printoptions(precision=10)
torch.set_float32_matmul_precision('medium')
    

# Add the directory containing the module to sys.path
import sys, pathlib
root_path = pathlib.Path(__file__).resolve().parents[2]
repo_path = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(repo_path))
# from datasets.dataset_utils import specific_radius_mfp_loader
# from models.optional_input_ranked_transformer import OptionalInputRankedTransformer


# helper functions 
from data_process import  plot_NMR, convert_to_tensor_1d_nmr
from flask_utils import build_actual_response

from inference.inference_utils import inference_topK, get_inputs_and_indicators_from_NMR_tensors


app = Flask(__name__)
CORS(app)

from flask import Response
import requests
from concurrent.futures import ThreadPoolExecutor

@app.before_request
def basic_authentication():
    if request.method.lower() == 'options':
        return Response()
    
'''for a single model, show top-5'''
def show_topK(inputs, NMR_type_indicator, k=5, MW_range = None):
    retrievals_to_return = []
        
    returning_smiles, returning_names, returning_imgs, returning_MWs, returning_values =  inference_topK(
        inputs, NMR_type_indicator, model, rankingset_data, smiles_and_names, 
            k=k, mode = None, ground_truth_FP=None,
            similarity_mapping_showing = "both",
            filter_by_MW=MW_range,
            verbose=False,
            weighting_pred = None,
            infer_in_backend_service = True,
            img_size = 800
    )
    for i, (smile, name, (img_no_sim_map, image_with_sim_map), mw, cos_value) in enumerate(zip(returning_smiles, returning_names, returning_imgs, returning_MWs, returning_values)):
        
        retrievals_to_return.append({"smile": smile, 
                              "name": name, 
                              "MW": mw,
                              "cos": cos_value,
                              "image_no_sim_map": img_no_sim_map,
                              "image_with_sim_map": image_with_sim_map,
                                    })

        if i == k:
            break
        
    with ThreadPoolExecutor(max_workers=8) as executor:
        np_results = list(executor.map(fetch_np, [entry['smile'] for entry in retrievals_to_return]))

    # Step 4: Merge the results into the retrievals
    for entry, np_data in zip(retrievals_to_return, np_results):
        entry["np_class"] = np_data

    return retrievals_to_return



def fetch_np(smile):
    try:
        np_url = f"https://npclassifier.gnps2.org/classify?smiles={smile}"
        res = requests.get(np_url, timeout=5)
        return res.json()
    except Exception as e:
        print(f"[WARNING] NPClassifier failed for {smile}: {e}")
        return {"error": "fetch_failed"}
    
    
@app.route('/api/hello', methods=['GET','OPTIONS'])
def hello():
    return build_actual_response(jsonify({'retrievals': "hello"}))

@app.route('/api/generate-retrievals', methods=['POST','OPTIONS'])
def generate_image():
    data = request.get_json()
    print("bakend received")
  
    HSQC = data['HSQC']
    C_NMR = data['C_NMR']
    H_NMR = data['H_NMR']
    mw = data['MW']
    mw_range = data['MW_range']
    HSQC_format = data['HSQC_format']
    k = data['k_samples']
    # print("HSQC_format", HSQC_format)
    
    '''
    option 1 means the input is in the format of "h, c, peak_sign" 
    option 2 means the input is in the format of "c, h, peak_sign"
    '''
    swap_c_h = HSQC_format == "option1"
    
    # convert input strings to tensor
    if HSQC == "":
        hsqc = None
    else:
        hsqc_stacked = []
        for peak in HSQC.split("\n"):
            if peak.strip() == "":
                continue
            hsqc_peak = np.fromstring(peak, sep=", ")
            if len(hsqc_peak) == 2:
                hsqc_peak = np.append(hsqc_peak, 0)
            hsqc_stacked.append(hsqc_peak)
        # print(hsqc_stacked)
        hsqc = torch.tensor(np.array(hsqc_stacked)).float()  
        if swap_c_h:         
            hsqc[:,[0,1]] = hsqc[:,[1,0]]
        # if no_peak_signs: # deprecated
        #     hsqc = torch.cat([hsqc, torch.ones(len(hsqc)).unsqueeze(1)], dim=1)
            
    c_tensor = None if C_NMR=="" else convert_to_tensor_1d_nmr(C_NMR, type="C-NMR") 
    h_tensor = None if H_NMR=="" else convert_to_tensor_1d_nmr(H_NMR, type="H-NMR")
    
    NMR_type_indicator, inputs = get_inputs_and_indicators_from_NMR_tensors(
        include_hsqc = hsqc is not None, 
        include_c_nmr = c_tensor is not None, 
        include_h_nmr = h_tensor is not None,
        include_MW = mw != "",
        hsqc = hsqc, c_tensor = c_tensor, h_tensor = h_tensor, mw = float(mw) if mw != "" else None,
    )

    retrieved_molecules = show_topK(inputs, NMR_type_indicator, k=k, MW_range = mw_range)
    nmr_fig_str= plot_NMR(hsqc, c_tensor, h_tensor)
    res = jsonify({'retrievals': retrieved_molecules, "NMR_plt": nmr_fig_str})
    return build_actual_response(res)


@app.route('/api/search-retrievals-by-smiles', methods=['POST','OPTIONS'])
def search_retrievals():
    data = request.get_json()
  
    SMILES = data['SMILES']
    k = data['k_samples']
    FP = fp_loader.build_mfp_for_new_SMILES(SMILES).unsqueeze(0).to("cuda")
    topk = retrieve_top_k_by_rankingset(rankingset_data, FP, smiles_and_names, k=k)
    returning_smiles, returning_names, returning_imgs, returning_MWs, returning_values = return_infos_from_topk(topk, FP)
    
    retrievals_to_return = []
    for i, (smile, name, (img_no_sim_map, image_with_sim_map), mw, cos_value) in enumerate(zip(returning_smiles, returning_names, returning_imgs, returning_MWs, returning_values)):
        
        retrievals_to_return.append({"smile": smile, 
                              "name": name, 
                              "MW": mw,
                              "cos": cos_value,
                              "image_no_sim_map": img_no_sim_map,
                              "image_with_sim_map": image_with_sim_map,
                                    })

    res = jsonify({'retrievals': retrievals_to_return})
    return build_actual_response(res)
    
if __name__ == '__main__':
    from waitress import serve
    # step 1: load model and datamodule   (here we assume we use all three NMRs)
    from datasets.dataset_utils import  fp_loader_configer
    from inference.inference_utils import choose_model, return_infos_from_topk, retrieve_top_k_by_rankingset

    fp_loader_configer.select_version("Hash_Entropy")
    fp_loader = fp_loader_configer.fp_loader
    
    hparams, model = choose_model("backend", return_data_loader=False)

    print("model device: ", model.device)

    # step 2: load rankingset
    smiles_and_names = pickle.load(open(f'{root_path}/inference/inference_metadata_name_updated.pkl', 'rb'))
    rankingset_path = f'{root_path}/inference/non_collision_FP_rankingset_r6_dim_16384/FP.pt'
    rankingset_data = torch.load(rankingset_path).to("cuda")
    # smiles_to_NMR_path = pickle.load(open(f'{root_path}/inference/SMILES_chemical_to_NMR_paths.pkl','rb'))

    print("starting server")
    # serve(app, host="0.0.0.0", port=6660)
    app.run(host='0.0.0.0', port=6660)
