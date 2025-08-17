from rdkit import Chem
import os

input_files = [
    "code_final/datasets/zuhe/data_before/npass.txt",
    "code_final/datasets/zuhe/data_before/tcmbank.txt",
    "code_final/datasets/zuhe/data_before/inflamnat.txt",
    "code_final/datasets/zuhe/data_before/cmaup.txt"
]

output_dir = "code_final/datasets/zuhe/data_remove_invalid/"
os.makedirs(output_dir, exist_ok=True)

for input_file in input_files:
    
    base_name = os.path.basename(input_file)           # npass.txt
    new_name = base_name.replace(".txt", "_after.txt") # npass_after.txt
    output_file = os.path.join(output_dir, new_name)   # datasets/zuhe/data_remove_invalid/npass_after.txt

    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            smiles = line.strip()
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                outfile.write(smiles + "\n")