import dash, dash_bootstrap_components as dbc

from dash.dependencies import Input, Output, State
from dash import dcc, html
from components.maindash import app
from components.constants.example_data import example_hsqc

@app.callback(
    Output('outputs', 'children'),
    [Input('model_path_button', 'n_clicks')],
    State('model_path', 'value'),
    State('hsqc_in', 'value'),
)
def update_model_outputs(n_clicks, path_value, hsqc_raw):
  if n_clicks is None and path_value is None and hsqc_raw is None:
    return None
  return None


app_layout = html.Div([
    html.H1('SMART 4.5'),
    dbc.Row([
        dbc.Col([
            dbc.Input(id = "model_path", type="text")
        ], width=8),
        dbc.Col([
            dbc.Button("Go!", id="model_path_button", color="primary", className="w-100")
        ], width=4)
    ]),
    dbc.Row([
      dbc.Col([
        dbc.Textarea(id = "hsqc_in", rows = 15, value=example_hsqc)
      ], width=6)
    ]),
    dbc.Row(id="outputs")
], className="container")
