from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib.pyplot as plt
import io
from PIL import Image
import numpy as np

def highlight_fragments(main_smiles, fragment_smiles_list):
    """
    Draw a molecule with different fragments highlighted
    
    Parameters:
    main_smiles (str): SMILES string of the main molecule
    fragment_smiles_list (list): List of SMILES strings of fragments to highlight
    
    Returns:
    list: List of PIL Image objects with highlighted fragments
    """
    # Convert main SMILES to RDKit molecule
    main_mol = Chem.MolFromSmiles(main_smiles)
    if main_mol is None:
        raise ValueError(f"Invalid SMILES string: {main_smiles}")
    
    # Add 2D coordinates to the molecule
    AllChem.Compute2DCoords(main_mol)
    
    images = []
    
    # Process each fragment
    for i, frag_smiles in enumerate(fragment_smiles_list):
        # Convert fragment SMILES to RDKit molecule
        frag_mol = Chem.MolFromSmiles(frag_smiles)
        if frag_mol is None:
            print(f"Warning: Invalid fragment SMILES: {frag_smiles}")
            continue
            
        # Find the substructure match
        matches = main_mol.GetSubstructMatches(frag_mol)
        
        if not matches:
            print(f"Fragment {frag_smiles} not found in the main molecule")
            continue
        
        # Create atom and bond highlights
        atoms_to_highlight = set()
        bonds_to_highlight = set()
        
        for match in matches:
            # Add atoms to highlight
            atoms_to_highlight.update(match)
            
            # Add bonds to highlight
            for bond_idx in range(frag_mol.GetNumBonds()):
                frag_bond = frag_mol.GetBondWithIdx(bond_idx)
                begin_atom = match[frag_bond.GetBeginAtomIdx()]
                end_atom = match[frag_bond.GetEndAtomIdx()]
                
                for main_bond_idx in range(main_mol.GetNumBonds()):
                    main_bond = main_mol.GetBondWithIdx(main_bond_idx)
                    if (main_bond.GetBeginAtomIdx() == begin_atom and 
                        main_bond.GetEndAtomIdx() == end_atom) or \
                       (main_bond.GetBeginAtomIdx() == end_atom and 
                        main_bond.GetEndAtomIdx() == begin_atom):
                        bonds_to_highlight.add(main_bond_idx)
                        break
        
        # Draw the molecule with highlighted fragments
        drawer = rdMolDraw2D.MolDraw2DCairo(500, 500)
        drawer.drawOptions().prepareMolsBeforeDrawing = False
        
        # Define colors for highlighting
        atom_colors = {atom_idx: (0.7, 0.0, 0.0) for atom_idx in atoms_to_highlight}
        bond_colors = {bond_idx: (0.7, 0.0, 0.0) for bond_idx in bonds_to_highlight}
        
        drawer.DrawMolecule(main_mol, highlightAtoms=list(atoms_to_highlight), 
                          highlightBonds=list(bonds_to_highlight),
                          highlightAtomColors=atom_colors,
                          highlightBondColors=bond_colors)
        drawer.FinishDrawing()
        
        # Convert to PIL Image
        png_data = drawer.GetDrawingText()
        image = Image.open(io.BytesIO(png_data))
        images.append(image)
        
    return images

def display_images(images, fragment_smiles_list):
    """Display all images in a grid"""
    n_images = len(images)
    if n_images == 0:
        print("No images to display")
        return
    
    # Calculate grid dimensions
    cols = min(3, n_images)
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, (img, frag_smiles) in enumerate(zip(images, fragment_smiles_list)):
        if i < len(axes):
            axes[i].imshow(np.array(img))
            axes[i].set_title(f"Fragment: {frag_smiles}")
            axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(n_images, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.show()

# Example usage
def main():
    # Example molecule (Celecoxib)
    main_smiles = "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)CF"
    
    # Example fragments
    fragment_smiles_list = [
        "S(=O)(=O)N",     # Sulfonamide group
        "CC1=CC=C(C=C1)",  # Tolyl group
        "C2=CC(=NN2)",    # Pyrazole core
        "CF"              # Fluoromethyl group
    ]
    
    # Generate and display the highlighted images
    images = highlight_fragments(main_smiles, fragment_smiles_list)
    display_images(images, fragment_smiles_list)

if __name__ == "__main__":
    main()