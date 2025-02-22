import  requests, json, pickle, tqdm, os

def get_superclass_and_glycoside(smiles):
    try:
        # smiles = 'CC1C(O)CC2C1C(OC1OC(COC(C)=O)C(O)C(O)C1O)OC=C2C(O)=O'
        url = f"https://npclassifier.gnps2.org/classify?smiles={smiles}"
        response = requests.get(url)
        json_dat = json.loads(response.content)
        superclass_results = json_dat['superclass_results']
        isglycoside = json_dat['isglycoside']
        if len(superclass_results) == 0:
            superclass_results = ["unknown"]
        return superclass_results, isglycoside
    except Exception as e:
        print(f"Error in {smiles}")
        print(e)
        return ["unknown"], None