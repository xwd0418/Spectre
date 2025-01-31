import torch
import torch.nn.functional as F
import torch, numpy as np
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

def retrieve_by_rankingset(data, prediction_query, smiles_and_names):
    query = F.normalize(prediction_query, dim=1, p=2.0).squeeze()

    results = []
    query_products = (data @ query)
    values, indices = torch.topk(query_products,k=1000)
    # instead of topk, we will sort the values and get all
    # values, indices = torch.sort(query_products, descending=True)
    
    for value, idx in zip(values, indices):
        results.append((value, idx, data[idx].nonzero()))
                        
    ret = [(value, smiles_and_names[i], fp) for value, i, fp in results]
    # print(torch.tensor(idx))
    # retrieved_FP = [all_fp[i] for i in idx]
    # print(results[0])
  
    return ret

def get_delimeter(delimeter_name):
    match delimeter_name:
        case "HSQC_start":
            return torch.tensor([-1,-1,-1]).float()
        case "HSQC_end":
            return torch.tensor([-2,-2,-2]).float()
        case "C_NMR_start":
            return torch.tensor([-3,-3,-3]).float()
        case "C_NMR_end":
            return torch.tensor([-4,-4,-4]).float()
        case "H_NMR_start":
            return torch.tensor([-5,-5,-5]).float()
        case "H_NMR_end":
            return torch.tensor([-6,-6,-6]).float()
        case "solvent_start":
            return torch.tensor([-7,-7,-7]).float()
        case "solvent_end":
            return torch.tensor([-8,-8,-8]).float()
        case "ms_start":
            return torch.tensor([-12,-12,-12]).float()
        case "ms_end":
            return torch.tensor([-13,-13,-13]).float()
        case _:
            raise Exception(f"unknown {delimeter_name}")


def build_input(hsqc=None, c_tensor=None, h_tensor=None, mw = 457.0, mode = "Default"):
    '''
    hsqc, c_tensor, h_tensor are either None or torch.tensor
    will return the stacked input tensor as transformer input
    '''
    
    if hsqc is not None:
        if mode == "no_sign":
            hsqc = torch.abs(hsqc)
        elif mode == "flip_sign":
            hsqc[:,2] = -hsqc[:,2]
            
    if hsqc is None:
        hsqc = torch.empty(0,3)
          
    if c_tensor is None: 
        c_tensor = torch.empty(0,3)
    if h_tensor is None:
        h_tensor = torch.empty(0,3)
    
    inputs = torch.vstack([
                    get_delimeter("HSQC_start"),  hsqc,     get_delimeter("HSQC_end"),
                    get_delimeter("C_NMR_start"), c_tensor, get_delimeter("C_NMR_end"), 
                    get_delimeter("H_NMR_start"), h_tensor, get_delimeter("H_NMR_end"),
                    ])   

    mol_weight = torch.tensor([mw,0,0]).float()
    # print(inputs)
    return torch.vstack([inputs, get_delimeter("ms_start"), mol_weight, get_delimeter("ms_end")])

def convert_to_tensor_1d_nmr(tensor_1d):
    try:
        if tensor_1d == "":
            tensor_1d = None
        else:
            c_vals = np.fromstring(tensor_1d, sep=",")
            tensor_1d = np.zeros((c_vals.shape[0], 3))
            tensor_1d[:,0] = c_vals
            tensor_1d = torch.tensor(tensor_1d).float()
        return tensor_1d
    except Exception as e:
        raise e
    
def unpack_inputs(inputs):
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
    h_tensor = inputs[0,h_nmr_start:h_nmr_end,0]
    return hsqc, c_tensor, h_tensor

# unpack_inputs(inputs)

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
        ax1.scatter(pos[:,1], pos[:,0], c="blue", label="CH or CH3", s=5)
        ax1.scatter(neg[:,1], neg[:,0], c="red", label="CH2", s=5)
        # print("scatter!!")
        # print(pos, neg)
    ax1.set_title("HSQC")
    ax1.set_xlabel('Proton Shift (1H)')  # X-axis label
    ax1.set_xlim([0, 12])
    ax1.set_ylim([0, 220])
    ax1.invert_yaxis()
    ax1.invert_xaxis()
    ax1.legend()


    ax2 = fig.add_subplot(gs[1, 0])  # Smaller subplot
    if c_tensor is not None:
        ax2.scatter( torch.ones(len(c_tensor)), c_tensor[:,0], c="black", s=2)
    ax2.set_ylim([0, 220])
    ax2.set_title("13C-NMR")
    ax2.set_ylabel('Carbon Shift (13C)')
    ax2.set_xticks([])
    ax2.invert_yaxis()
    ax2.invert_xaxis()

    ax3 = fig.add_subplot(gs[0, 1])  # Smaller subplot
    if h_tensor is not None:
        ax3.scatter(h_tensor[:,0], torch.ones(len(h_tensor)),c="black", s=2)
    ax3.set_xlim([0, 12])
    ax3.set_title("1H-NMR")
    ax3.set_yticks([])
    ax3.invert_yaxis()
    ax3.invert_xaxis()

    # Adjust layout to prevent overlapping
    plt.tight_layout()

    # Save the plot to a BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Encode the BytesIO object to a base64 string
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        
    return img_base64

