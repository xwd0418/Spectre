
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem import rdFingerprintGenerator

smiles = "OCC1OC(OC2C(CO)OC(OC3C(CO)OC(OC4C(CO)OC(O)C(O)C4O)C(O)C3O)C(O)C2O)C(O)C(O)C1O"
radius = 3
mol = Chem.MolFromSmiles(smiles)
mol = Chem.AddHs(mol)

bitinfo = {}
fp = rdMolDescriptors.GetMorganFingerprint(mol, radius=radius, bitInfo=bitinfo)
x1 = str(sorted(bitinfo.values()))


gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius)
ao = rdFingerprintGenerator.AdditionalOutput()
ao.AllocateBitInfoMap()

fp = gen.GetFingerprint(Chem.MolFromSmiles("c1ccccc1"), additionalOutput=ao)
fp = gen.GetFingerprint(Chem.MolFromSmiles("CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"), additionalOutput=ao)
fp = gen.GetFingerprint(Chem.MolFromSmiles("Cc1ccccc1"), additionalOutput=ao)



fp = gen.GetFingerprint(mol, additionalOutput=ao)
x2 = str(sorted(ao.GetBitInfoMap().values()))

print(x1 == x2)