import math
import torch
from torch.utils.data import Dataset
from utils import SmilesEnumerator
import numpy as np
import re

class SmileDataset(Dataset):

    def __init__(self, args, data, content, block_size):
        chars = sorted(list(set(content)))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d smiles, %d unique characters.' % (data_size, vocab_size))
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.max_len = block_size
        self.vocab_size = vocab_size
        self.data = data
        self.debug = args.debug
        self.tfm = SmilesEnumerator()
    
    def __len__(self):
        if self.debug:
            return math.ceil(len(self.data) / (self.max_len + 1))
        else:
            return len(self.data)

    def __getitem__(self, idx):
        smiles = self.data[idx]  
        smiles = smiles.strip()
        pattern =  "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)
        smiles += str('<')*(self.max_len - len(regex.findall(smiles)))
        if len(regex.findall(smiles)) > self.max_len:
            smiles = smiles[:self.max_len]
        smiles=regex.findall(smiles)
        dix =  [self.stoi[s] for s in smiles]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y