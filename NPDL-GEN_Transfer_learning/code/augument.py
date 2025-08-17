import numpy as np
from rdkit import Chem

def randomize_smiles(smiles, canonical=False, isomericSmiles=True):
    """Perform a randomization of a SMILES string
    must be RDKit sanitizable"""
    m = Chem.MolFromSmiles(smiles)
    ans = list(range(m.GetNumAtoms()))
    np.random.shuffle(ans)
    nm = Chem.RenumberAtoms(m, ans)
    return Chem.MolToSmiles(nm, canonical=canonical, isomericSmiles=isomericSmiles)
if __name__ == '__main__':

    with open('code_final/datasets/inflamnat_after_filter.txt', 'r') as file:
        smiles_list = file.readlines()

  
    augmented_smiles_list = []
  
  
    for smiles in smiles_list:
        original_smiles = smiles.strip()
        augmented_smiles_list.append(original_smiles + '\n')
        for _ in range(9):  
            augmented_smiles_list.append(randomize_smiles(original_smiles) + '\n')

   
    with open('code_final/datasets/inflamnat_after_filter_augument_9.txt', 'w') as file:
        file.writelines(augmented_smiles_list)
