import base64
from io import BytesIO
import pickle
import dash
import dash_bootstrap_components as dbc
from dash import html

load_path = "/workspace/smart4.5/ignore/dump.pkl"
with open(load_path, "rb") as f:
  obj = pickle.load(f)

app = dash.Dash(
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)
# https://stackoverflow.com/a/67154617
def pil_to_b64(im, enc_format="png", **kwargs):
    """
    Converts a PIL Image into base64 string for HTML displaying
    :param im: PIL Image object
    :param enc_format: The image format for displaying. If saved the image will have that extension.
    :return: base64 encoding
    """

    buff = BytesIO()
    im.save(buff, format=enc_format, **kwargs)
    encoded = base64.b64encode(buff.getvalue()).decode("utf-8")

    return encoded



def build_ul(items):
  return html.Ul(
    [html.Li(i) for i in items]
  )
def molecule_summary(img, fp, smiles, rank):
  return dbc.Row([
    dbc.Col(html.H5(f"Rank {rank}"), width=12),
    dbc.Col(html.Img(src="data:image/png;base64, " + pil_to_b64(img), className="w-100"), width=4),
    dbc.Col([
      html.P(f"Fingerprint: {fp}"),
      html.P(f"SMILES: {smiles}"),
    ], width=8),
  ], className="border border-dark border-3")
def build_rank_results(obj):
  children = []
  for i, (sm_l, im_l, fp_l) in enumerate(zip(obj["smiles"], obj["imgs"], obj["fps"])):
    sub_children = []
    for sm, im, fp in zip(sm_l, im_l, fp_l):
      sub_children.append(molecule_summary(im, fp, sm, i))
    children.append(dbc.Container(sub_children, className="border border-danger border-3 my-1"))
  return dbc.Container(children, className="border border-primary border-3")

def build_layout(obj):
  components = []
  for i, (k,v) in enumerate(obj.items()): # k: id, v: object from ranking_visuzalization.ipynb
    components.append(html.H3(f"Sample {i}, Molecule {k}, Rank: {v['rank']}, Smiles {v['single_smiles']}"))
    desc = []
    desc.append(html.P(f"Label Fingerprint: {v['label_fp']}"))
    desc.append(html.P(f"Predicted Fingerprint: {v['out_fp']}"))
    desc.append(html.P(f"All SMILES with this FP"))
    desc.append(build_ul(v['original_smiles']))
    row = dbc.Row([
      dbc.Col(html.Img(src="data:image/png;base64, " + pil_to_b64(v['base_image']), className="w-100"), width=4),
      dbc.Col(desc, width=8),
    ])
    components.append(row)
    components.append(build_rank_results(v))
    
  return dbc.Container(components, fluid=True)

app.layout = build_layout(obj)

if __name__ == "__main__":
    app.run_server(debug=True)
