from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import IsoSpecPy
import re
from collections import defaultdict
import pickle
import os
import torch
from tqdm import tqdm

DATA_ROOT = '/workspace/SMILES_dataset'

def smiles_to_formula(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    formula = rdMolDescriptors.CalcMolFormula(mol)
    
    return parse_charge(formula)[0]

def get_isotopic_distribution(formula: str, threshold: float = 0.99):
    iso = IsoSpecPy.IsoTotalProb(formula = formula, prob_to_cover=threshold)
    distribution = []
    
    for conf in iso:
        mass, prob = conf
        distribution.append((mass, prob))

    return distribution

def parse_charge(raw_formula: str):
    match = re.match(r"^([A-Za-z0-9]+)([+-]\d*)?$", raw_formula.strip())
    if not match:
        raise ValueError(f"Invalid formula: {raw_formula}")
    
    formula = match.group(1)
    charge_str = match.group(2)

    if charge_str:
        if charge_str in ('+', '-'):
            charge = 1 if charge_str == '+' else -1
        else:
            charge = int(charge_str)
    else:
        charge = 0

    return formula, charge

if __name__ == '__main__':
    for split in ('train', 'val', 'test'):
        os.makedirs(os.path.join(DATA_ROOT, split, 'IsotopicDistribution'), exist_ok=True)
        with open(os.path.join(DATA_ROOT, split, 'SMILES/index.pkl'), 'rb') as f:
            all_smiles: dict[int, str] = pickle.load(f)

        iso_dist = defaultdict(list)
        for i, smiles in tqdm(all_smiles.items()):
            formula = smiles_to_formula(smiles)
            distribution = get_isotopic_distribution(formula, threshold=0.99)
            
            for mass, prob in distribution:
                iso_dist[i].append((mass, prob))
            with open(os.path.join(DATA_ROOT, split, 'IsotopicDistribution', f'{i}.pt'), 'wb') as f:
                torch.save(torch.tensor(iso_dist[i], dtype=torch.float64), f)