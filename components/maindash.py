import dash, flask, dash_bootstrap_components as dbc

server = flask.Flask('app')
app = dash.Dash('app', server=server, external_stylesheets=[dbc.themes.BOOTSTRAP])