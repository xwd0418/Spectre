from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

# Input SMILES
smiles = "CCOCC"

# Define the radius
radius = 2

def get_sub_structures(smiles, radius):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")

    # Get atom environment hashes
    atom_envs = rdMolDescriptors.GetMorganFingerprint(mol, radius).GetNonzeroElements()
    print(atom_envs)
    substructures = []
    for atom_idx, count in atom_envs.items():
        # Find atom environment (use correct 0-based indexing)
        print(mol, radius, atom_idx)
        env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
        amap = {}  # Atom map to track indices
        submol = Chem.PathToSubmol(mol, env, atomMap=amap)
        sub_smiles = Chem.MolToSmiles(submol)
        substructures.append((sub_smiles, count))

    return substructures
        

smiles = "CCOCC"
radius = 2
substructures = get_sub_structures(smiles, radius)

print("Substructures based on Morgan algorithm:")
for sub_smiles, count in substructures:
    print(f"Substructure: {sub_smiles}, Count: {count}")