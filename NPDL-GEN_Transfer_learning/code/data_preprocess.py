from rdkit import Chem
def filter_mol(mol, max_heavy_atoms=100, min_heavy_atoms=10):
    """Filters molecules on number of heavy atoms"""
    if mol is not None:
        num_heavy = min_heavy_atoms<mol.GetNumHeavyAtoms()<max_heavy_atoms
        if num_heavy:
            return True
        else:
            return False


with open("code_final/datasets/zuhe/merged_smiles.txt", "r") as infile:
    with open("code_final/datasets/zuhe/merged_smiles_after.txt", "w") as outfile:
        lines = infile.readlines()
        for line in lines:
            line = line.strip()
            mol = Chem.MolFromSmiles(line)
            if filter_mol(mol):
                outfile.write(line + "\n")