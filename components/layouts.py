import dash
import dash_bootstrap_components as dbc

from dash.dependencies import Input, Output, State
from dash import dcc, html
from components.maindash import app
from components.constants.example_data import example_hsqc
from components.helpers.parse_hsqc import parse_hsqc
from components.helpers.slow_forward import fwomp, fwomp_rank
from components.helpers.build_rank_res import build

@app.callback(
    Output('loading-outputs', 'children'),
    [Input('model_path_button', 'n_clicks')],
    State('model_path', 'value'),
    State('hsqc_in', 'value'),
)
def update_model_outputs(n_clicks, path_value, hsqc_raw):
  if n_clicks is None and path_value is None and hsqc_raw is None:
    return None
  print("Doing a Forward Pass")
  try:
    float_hsqc = parse_hsqc(hsqc_raw)
    print("Successfully Parsed")
    fp_pred = fwomp(float_hsqc)
    print("Successfully Forwarded")
    t10_ranks = fwomp_rank(fp_pred[0])
    out = build(t10_ranks, "tempdata/SMILES_dataset/val", identifier = path_value)
    return html.Div(out)
  except Exception as err:
    return f"Failed to perform Forward. {err=}"


app_layout = html.Div([
    html.H1('SMART 4.5'),
    dbc.Row([
        dbc.Col([
            dbc.Input(id="model_path", type="text")
        ], width=8),
        dbc.Col([
            dbc.Button("Go!", id="model_path_button",
                       color="primary", className="w-100")
        ], width=4)
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Textarea(id="hsqc_in", rows=15)
        ], width=6)
    ]),
    dcc.Loading(
        id="loading",
        children=[html.Div(id="loading-outputs")],
        type="circle",
    )
], className="container")
