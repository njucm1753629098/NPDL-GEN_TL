def extract_smiles(file_path):
    smiles_set = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                smiles = line.strip()
                if smiles:
                    smiles_set.add(smiles)
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='ISO-8859-1') as f:
            for line in f:
                smiles = line.strip()
                if smiles:
                    smiles_set.add(smiles)
    return smiles_set

file_paths = ['code_final/datasets/zuhe/data_remove_invalid/tcmbank_after.txt', 'code_final/datasets/zuhe/data_remove_invalid/npass_after.txt', 'code_final/datasets/zuhe/data_before/inflamnat.txt', 'code_final/datasets/zuhe/data_before/cmaup.txt']

unique_smiles = set()
for file_path in file_paths:
    unique_smiles.update(extract_smiles(file_path))

with open('code_final/datasets/zuhe/merged_smiles.txt', 'w') as f:
    for smiles in unique_smiles:
        f.write(smiles + '\n')