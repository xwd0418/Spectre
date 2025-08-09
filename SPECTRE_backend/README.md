# To run the backend service
`cd SPECTRE_backend`
`RUNNING_BACKEND=1 python3 app.py`

# Backend main file: app.py
This file is to initialize and run the server. The initialization includes loading models and rankingset(also referred to as the retrieval set).

**app.py** provides 3 APIs
* @app.route('/api/hello', methods=['GET','OPTIONS'])
    - connection testing
* @app.route('/api/generate-retrievals', methods=['POST','OPTIONS'])
    - retrieval generation, the single most important API in the web service
* @app.route('/api/search-retrievals-by-smiles', methods=['POST','OPTIONS'])
    - retrieval set search by SMILES string

# data_process.py
A file of helper functions: data formatting, NMR data plotting, etc. 

# Other files
Not important
