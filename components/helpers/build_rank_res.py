import torch, pickle, base64, io
import dash_bootstrap_components as dbc

from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

from dash import html

# https://stackoverflow.com/questions/60712647/displaying-pil-images-in-dash-plotly
def pil_to_b64(im, enc_format="png", **kwargs):
  """
  Converts a PIL Image into base64 string for HTML displaying
  :param im: PIL Image object
  :param enc_format: The image format for displaying. If saved the image will have that extension.
  :return: base64 encoding
  """

  buff = io.BytesIO()
  im.save(buff, format=enc_format, **kwargs)
  encoded = base64.b64encode(buff.getvalue()).decode("utf-8")

  return encoded

def build(idxs, path, identifier = None):
  features = {}
  data_path = Path(path)
  name_folder_path = "Chemical"
  smiles_folder_path = "SMILES"

  with open(data_path / name_folder_path / "index.pkl", "rb") as f:
    chemical_index = pickle.load(f)
  with open(data_path / smiles_folder_path / "index.pkl", "rb") as f:
    smiles_index = pickle.load(f)
  for v in idxs:
    chemical = chemical_index[v]
    smiles = smiles_index[v]
    features[v] = {
      "Chemical": chemical,
      "SMILES": smiles
    }
  
  out = []

  for i, v in enumerate(idxs):
    mol = Chem.MolFromSmiles(features[v]["SMILES"])
    im = Chem.Draw.MolToImage(mol, size=(200, 300))

    label_tag = html.H3(f"Rank {i + 1}, Sample {v}")
    smiles_tag = html.P(features[v]["SMILES"])
    chemical_tag = html.P(features[v]["Chemical"])
    img_tag = html.Img(src="data:image/png;base64, " + pil_to_b64(im))

    out.append(dbc.Col([label_tag, chemical_tag, smiles_tag, img_tag], className="border border-primary border-5 m-4", width=4))
  
  try:
    if identifier:
      mol = Chem.MolFromSmiles(identifier)
      im = Chem.Draw.MolToImage(mol, size=(200, 300))

      label_tag = html.H3(f"Query")
      smiles_tag = html.P(identifier)
      img_tag = html.Img(src="data:image/png;base64, " + pil_to_b64(im))

      identifier_obj = dbc.Col([label_tag, smiles_tag, img_tag], className="border border-warning border-5 m-4", width=4)

      return dbc.Container(identifier_obj), dbc.Container(out)
  except:
    pass
  return dbc.Container(out)
