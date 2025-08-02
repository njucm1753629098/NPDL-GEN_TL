from utils import check_novelty, sample, canonic_smiles
#from dataset import SmileDataset
from rdkit.Chem import QED
from rdkit.Chem import Crippen
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit import Chem
import math
from tqdm import tqdm
import argparse
from model import GPT, GPTConfig
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import get_mol
import re
from rdkit.Chem import Descriptors
import json
from rdkit.Chem import RDConfig
import json
import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
from rdkit.Chem.rdMolDescriptors import CalcTPSA

if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_weight', type=str, help="path of model weights", required=True)
        parser.add_argument('--csv_name', type=str, help="name to save the generated mols in csv format", required=True)
        parser.add_argument('--batch_size', type=int, default = 512, help="batch size", required=False)
        parser.add_argument('--gen_size', type=int, default = 10000, help="number of molecules to generate", required=False)
        parser.add_argument('--vocab_size', type=int, default = 144, help="vocab size", required=False) 
        parser.add_argument('--block_size', type=int, default = 212, help="block size", required=False)  
        parser.add_argument('--n_layer', type=int, default = 12, help="number of layers", required=False)
        parser.add_argument('--n_head', type=int, default = 12, help="number of heads", required=False)
        parser.add_argument('--n_embd', type=int, default = 768, help="embedding dimension", required=False)

        args = parser.parse_args()

        # starting character for generation
        context = "C"   

        # vocabulary
        content = ['#', '%10', '%11', '%12', '%13', '%14', '%15', '(', ')', '-', '.', '/', '1', '2', '3', '4', '5', '6', '7', '8', '9', '<', '=', 'B', 'Br', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S', '[*]', '[11CH3]', '[11C]', '[13CH2]', '[13CH3]', '[13CH]', '[13C]', '[13cH]', '[13c]', '[14CH2]', '[14CH3]', '[14CH]', '[14C]', '[14cH]', '[14c]', '[15N]', '[15n]', '[18OH]', '[18O]', '[2H]', '[3H]', '[Ag+]', '[Al+3]', '[Al]', '[As]', '[B-]', '[B@-]', '[Bi+3]', '[Bi]', '[Br-]', '[C+]', '[C-]', '[C@@H]', '[C@@]', '[C@H]', '[C@]', '[CH-]', '[CH2+]', '[CH2-]', '[CH2]', '[CH]', '[C]', '[Ca+2]', '[Ca]', '[Cl-]', '[Co+2]', '[Co+3]', '[Co+]', '[Co]', '[Cu+2]', '[Cu]', '[Fe+2]', '[Fe+3]', '[Fe+5]', '[Fe]', '[Ga]', '[H+]', '[HH]', '[H]', '[Hg]', '[I+]', '[I-]', '[K+]', '[K]', '[Mg+2]', '[Mg]', '[Mn+2]', '[N+]', '[N-]', '[N@+]', '[N@@+]', '[N@@]', '[N@]', '[NH+]', '[NH-]', '[NH2+]', '[NH3+]', '[N]', '[Na+]', '[Na]', '[O+]', '[O-]', '[OH+]', '[OH-]', '[OH2+]', '[O]', '[P@@]', '[P@]', '[PH+]', '[Pb]', '[Pt]', '[S+]', '[S-]', '[S@@]', '[S@]', '[SH]', '[Se+]', '[Se]', '[Si]', '[Zn+2]', '[n+]', '[n-]', '[nH+]', '[nH]', '[o+]', '[s+]', '\\', 'c', 'n', 'o', 's']
        
        chars = sorted(list(set(content)))
        stoi = { ch:i for i,ch in enumerate(chars) }
        itos = { i:ch for i,ch in enumerate(chars) }
     
        pattern =  "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)
        
        mconf = GPTConfig(args.vocab_size, args.block_size,
                       n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd)
        model = GPT(mconf)


        model.load_state_dict(torch.load(args.model_weight))
        model.to('cuda')
        print('Model loaded')

        gen_iter = math.ceil(args.gen_size / args.batch_size)
            
            
        all_dfs = []
        all_metrics = []
        molecules = []
            
        for i in tqdm(range(gen_iter)):
            x = torch.tensor([stoi[s] for s in regex.findall(context)], dtype=torch.long)[None,...].repeat(args.batch_size, 1).to('cuda')
            y = sample(model, x, args.block_size, temperature=1, sample=True, top_k=None)   
            for gen_mol in y:
                completion = ''.join([itos[int(i)] for i in gen_mol])
                completion = completion.replace('<', '')
                # gen_smiles.append(completion)
                mol = get_mol(completion)
                if mol:
                    molecules.append(mol)
                "Valid molecules % = {}".format(len(molecules))

        mol_dict = []

        for i in molecules:
            mol_dict.append({'molecule' : i, 'smiles': Chem.MolToSmiles(i)})

        results = pd.DataFrame(mol_dict)
        canon_smiles = [canonic_smiles(s) for s in results['smiles']]
        unique_smiles = list(set(canon_smiles))
        novel_ratio = check_novelty(unique_smiles, 'NPDL-GEN&Transfer_learning/datasets/merged_smiles_after.txt')
          
        print('Valid ratio: ', np.round(len(results)/10000, 3))
        print('Unique ratio: ', np.round(len(unique_smiles)/len(results), 3))
        print('Novelty ratio: ', np.round(novel_ratio/100, 3))
        
        results['qed'] = results['molecule'].apply(lambda x: QED.qed(x) )
        all_dfs.append(results)
        results = pd.concat(all_dfs)
        results.to_csv('code_final/' + args.csv_name + '.csv', index = False)
        
        
             

         
