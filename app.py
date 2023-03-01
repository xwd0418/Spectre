from components.maindash import app
from components.layouts import app_layout

if __name__ == '__main__':
  app.layout = app_layout
  app.run_server(debug=True, port=8060)