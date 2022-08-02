import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

#FP used in this system uses different ranges for each radius to reduce bit collision. ie 0~2047 for radius 0, 2048~4095 for radius 1, and 4096~6143 for radius 2
def FP_generator(SMILES,radi): # radi is radius for morgan fingerprint
    binary = np.zeros((2048*(radi+1)), int)
    mol = Chem.MolFromSmiles(SMILES)
    mol_H = Chem.AddHs(mol)
    mol_bi_H = {}
    for r in range(radi+1):
        mol_fp_H = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol_H, radius=r, bitInfo=mol_bi_H, nBits = 2048)
        mol_bi_H_QC = []
        for i in mol_fp_H.GetOnBits():
            idx = mol_bi_H[i][0][0]
            radius_list = []
            for j in range(len(mol_bi_H[i])):
                atom_radi = mol_bi_H[i][j][1]
                radius_list.append(atom_radi) 
            atom = mol_H.GetAtomWithIdx(idx)
            symbol = atom.GetSymbol()
            neigbor = [x.GetAtomicNum() for x in atom.GetNeighbors()]
            if r in radius_list: #and symbol == 'C' and 1 in neigbor:#radius = 2, atom = Carbon, H possessed Carbon
                mol_bi_H_QC.append(i)
        bits = mol_bi_H_QC
        for i in bits:
            binary[(2048*r)+i] = 1
    return binary

def canonical(SMILES):
    from rdkit import Chem
    mol = Chem.MolFromSmiles(SMILES)
    smiles = Chem.MolToSmiles(mol, False) if mol is not None else None
    return smiles