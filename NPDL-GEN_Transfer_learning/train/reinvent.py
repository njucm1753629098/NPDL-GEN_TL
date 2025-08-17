import argparse
from utils import set_seed
import numpy as np
import wandb
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.cuda.amp import GradScaler
from model import GPT, GPTConfig
from trainer import Trainer, TrainerConfig
from dataset import SmileDataset
import math
from utils import SmilesEnumerator
import re
import os

if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--model_weight', type=str, help="path of model weights", required=True)
    parser.add_argument('--run_name', type=str,
                        help="name for wandb run", required=False)
    parser.add_argument('--debug', action='store_true',
                        default=False, help='debug')
    parser.add_argument('--vocab_size', type=int, default = 144, help="vocab size", required=False)
    parser.add_argument('--block_size', type=int, default = 212, help="block size", required=False)
    parser.add_argument('--n_layer', type=int, default=12,
                        help="number of layers", required=False)
    parser.add_argument('--n_head', type=int, default=12,
                        help="number of heads", required=False)
    parser.add_argument('--n_embd', type=int, default=768,
                        help="embedding dimension", required=False)
    parser.add_argument('--max_epochs', type=int, default=400,
                        help="total epochs", required=False)
    parser.add_argument('--batch_size', type=int, default=64,
                        help="batch size", required=False)
    parser.add_argument('--learning_rate', type=int,
                        default=6e-4, help="learning rate", required=False)


    args = parser.parse_args()
    set_seed(42)
    wandb.init(project="code_final", name=args.run_name)
    
    with open('NPDL-GEN_Transfer_learning/datasets/merged_smiles_after.txt','r') as file:
        smiles_list = file.readlines()
    smiles_list = [smiles.strip() for smiles in smiles_list]
     
    smiles = []
    with open('NPDL-GEN_Transfer_learning/datasets/train_merged_smiles.txt', 'r') as train_file:
        for line in train_file:
            smiles.append(line.strip())

    vsmiles = []
    with open('NPDL-GEN_Transfer_learning/datasets/test_merged_smiles.txt', 'r') as test_file:
        for line in test_file:
            vsmiles.append(line.strip())
    
    pattern = "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    max_len = 212
    print('Max len: ', max_len)

    smiles = [i + str('<')*(max_len - len(regex.findall(i.strip())))
                for i in smiles]
    vsmiles = [i + str('<')*(max_len - len(regex.findall(i.strip())))
                for i in vsmiles]
    '''
    whole_string = ' '.join(smiles + vsmiles)
    whole_string = sorted(list(set(regex.findall(whole_string))))
    print(whole_string)
    '''
    whole_string = ['#', '%10', '%11', '%12', '%13', '%14', '%15', '(', ')', '-', '.', '/', '1', '2', '3', '4', '5', '6', '7', '8', '9', '<', '=', 'B', 'Br', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S', '[*]', '[11CH3]', '[11C]', '[13CH2]', '[13CH3]', '[13CH]', '[13C]', '[13cH]', '[13c]', '[14CH2]', '[14CH3]', '[14CH]', '[14C]', '[14cH]', '[14c]', '[15N]', '[15n]', '[18OH]', '[18O]', '[2H]', '[3H]', '[Ag+]', '[Al+3]', '[Al]', '[As]', '[B-]', '[B@-]', '[Bi+3]', '[Bi]', '[Br-]', '[C+]', '[C-]', '[C@@H]', '[C@@]', '[C@H]', '[C@]', '[CH-]', '[CH2+]', '[CH2-]', '[CH2]', '[CH]', '[C]', '[Ca+2]', '[Ca]', '[Cl-]', '[Co+2]', '[Co+3]', '[Co+]', '[Co]', '[Cu+2]', '[Cu]', '[Fe+2]', '[Fe+3]', '[Fe+5]', '[Fe]', '[Ga]', '[H+]', '[HH]', '[H]', '[Hg]', '[I+]', '[I-]', '[K+]', '[K]', '[Mg+2]', '[Mg]', '[Mn+2]', '[N+]', '[N-]', '[N@+]', '[N@@+]', '[N@@]', '[N@]', '[NH+]', '[NH-]', '[NH2+]', '[NH3+]', '[N]', '[Na+]', '[Na]', '[O+]', '[O-]', '[OH+]', '[OH-]', '[OH2+]', '[O]', '[P@@]', '[P@]', '[PH+]', '[Pb]', '[Pt]', '[S+]', '[S-]', '[S@@]', '[S@]', '[SH]', '[Se+]', '[Se]', '[Si]', '[Zn+2]', '[n+]', '[n-]', '[nH+]', '[nH]', '[o+]', '[s+]', '\\', 'c', 'n', 'o', 's']
    train_dataset = SmileDataset(args, smiles, whole_string, max_len)
    valid_dataset = SmileDataset(args, vsmiles, whole_string, max_len)

    mconf = GPTConfig(args.vocab_size, args.block_size,  
                        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd)
    model = GPT(mconf)
    model.load_state_dict(torch.load(args.model_weight))
    model = torch.nn.DataParallel(model).cuda()

    tconf = TrainerConfig(max_epochs=args.max_epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
                            lr_decay=True, warmup_tokens=0.1*len(smiles)*max_len, 
                            final_tokens=args.max_epochs*len(smiles)*max_len, num_workers=10, 
                            ckpt_path=f'NPDL-GEN_Transfer_learning/weights/reinvent-gpt1-400.pt', block_size=train_dataset.max_len, generate=True)
    trainer = Trainer(model, train_dataset, valid_dataset,
                        tconf, train_dataset.stoi, train_dataset.itos)
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    df = trainer.train_reinvent(wandb)
