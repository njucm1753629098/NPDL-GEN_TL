import math
import logging
from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
from torch.cuda.amp import GradScaler
from utils import check_novelty, sample, canonic_smiles, get_mol, check_validity, to_tensor
import re
import pandas as pd
from rdkit import Chem
import csv
from rdkit.Chem import QED
logger = logging.getLogger(__name__)
from collections import Counter
 
        
class TrainerConfig:
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0 # for DataLoader

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)



class Trainer:
    def __init__(self, model, train_dataset, test_dataset, config, stoi, itos):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.prior = model
        self.agent = model
        # take over whatever gpus are on the system
        self.device = 'cpu'
        self.stoi = stoi
        self.itos = itos
        self.record = {'loss': [],
                       'prior_nll': [],
                       'agent_nll': []}
       
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = self.model.to(self.device)
            self.prior = self.prior.to(self.device)
            self.agent = self.agent.to(self.device)
            
    def save_checkpoint(self):
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info("saving %s", self.config.ckpt_path)
        torch.save(raw_model.state_dict(), self.config.ckpt_path)
        
    def train(self, wandb):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)
        scaler = GradScaler()
        def run_epoch(split):
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)
            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            for it, (x, y) in pbar:

                x = x.to(self.device)
                y = y.to(self.device)
                with torch.cuda.amp.autocast():
                    with torch.set_grad_enabled(is_train):
                        _, logits, loss, _ = model(x, y)
                        loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                        losses.append(loss.item())
                if is_train:
                    model.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    if config.lr_decay:
                        self.tokens += (y >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate
                    # report progress
                    wandb.log({'step_train_loss': loss, 'train_step': it + epoch*len(loader), 'learning_rate': lr})
                    pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")
            if is_train:
                return float(np.mean(losses))
            if not is_train:
                test_loss = float(np.mean(losses))
                logger.info("test loss: %f", test_loss)
                return test_loss
        best_loss = float('inf')
        self.tokens = 0 # counter used for learning rate decay
        molecules = []
        for epoch in range(config.max_epochs):
            train_loss = run_epoch('train')
            if self.test_dataset is not None:
                test_loss = run_epoch('test')
            wandb.log({'epoch_valid_loss': test_loss, 'epoch_train_loss': train_loss, 'epoch': epoch + 1})
            # supports early stopping based on the test loss, or just save always if no test set is provided
            good_model = self.test_dataset is None or test_loss < best_loss
            if self.config.ckpt_path is not None and good_model:
                best_loss = test_loss
                print(f'Saving at epoch {epoch + 1}')
                self.save_checkpoint()
        return None
        
    # AHC fine-tune with the first scoring function

    def train_ahc(self, wandb):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)
        scaler = GradScaler()
        pattern =  "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)
        context = "C"
        x = torch.tensor([self.stoi[s] for s in regex.findall(context)], dtype=torch.long)[None,...].repeat(64, 1).to('cuda')
        for step in tqdm(range(400), total=400):
            seqs, probs, log_probs = sample(self.agent, x, 200,temperature=0.8, sample=True, top_k=10)
            seqs_x = torch.tensor(seqs[:,:-1], dtype=torch.long)
            seqs_y = torch.tensor(seqs[:,1:], dtype=torch.long)
            
            agent_likelihood,_,_,_ = self.agent(seqs_x,seqs_y)
            prior_likelihood,_,_,_ = self.prior(seqs_x,seqs_y)
            smiles = []
            valid_smiles = []
            for seq in seqs: 
                completion = ''.join([self.itos[int(i)] for i in seq])
                completion = completion.replace('<', '')
                smiles.append(completion)
                if Chem.MolFromSmiles(completion) is not None:  
                    valid_smiles.append(completion)
            agent_likelihood = - agent_likelihood
            print("The size of agent_likelihood:",agent_likelihood.size())
            prior_likelihoood = - prior_likelihood
            print("The size of prior_likelihood:",prior_likelihood.size())
            
            filename = f"NPDL-GEN&Transfer_learning/score_results/iterations_ahc_gpt1_400_topk_0.25/step_{step}.csv"
            with open(filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['step_id', 'smiles', 'valid', 'qed'])  
                scores = []
                for s in smiles:
                    mol = Chem.MolFromSmiles(s)
                    is_valid = mol is not None  
                    qed_value = QED.qed(mol) if is_valid else 0  
                    scores.append(qed_value)
                    writer.writerow([step, s, is_valid, qed_value])  
            scores = torch.tensor(scores)
            print("The size of scores:",scores.size())
            scores = torch.autograd.Variable(scores).cuda()
            loss = self._compute_loss_ahc(prior_likelihood, agent_likelihood, scores)
           
            loss = loss.mean()
            model.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            scaler.step(optimizer)
            scaler.update()
            self.tokens = 0
            if config.lr_decay:
                self.tokens += (seqs >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
                if self.tokens < config.warmup_tokens:
                    # linear warmup
                    lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                else:
                    # cosine learning rate decay
                    progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                    lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                lr = config.learning_rate * lr_mult
                for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
            else:
                lr = config.learning_rate
            
            if self.config.ckpt_path is not None :
                print(f'Saving the model')
                self.save_checkpoint()
            
        return None
    
    '''
    # AHC fine-tune with the second scoring function 
    def train_ahc(self, wandb):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)
        scaler = GradScaler()
        pattern =  "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)
        context = "C"
        x = torch.tensor([self.stoi[s] for s in regex.findall(context)], dtype=torch.long)[None,...].repeat(64, 1).to('cuda')
        for step in tqdm(range(400), total=400):
            
            seqs, probs, log_probs = sample(self.agent, x, 200,temperature=0.8, sample=True, top_k=10) 
            seqs_x = torch.tensor(seqs[:,:-1], dtype=torch.long)
            seqs_y = torch.tensor(seqs[:,1:], dtype=torch.long)
            agent_likelihood,_,_,_ = self.agent(seqs_x,seqs_y)
            prior_likelihood,_,_,_ = self.prior(seqs_x,seqs_y)
            smiles = []
            valid_smiles = []
            for seq in seqs: 
                completion = ''.join([self.itos[int(i)] for i in seq])
                completion = completion.replace('<', '')
                smiles.append(completion)
                if Chem.MolFromSmiles(completion) is not None:  
                    valid_smiles.append(completion)
            agent_likelihood = - agent_likelihood
            prior_likelihoood = - prior_likelihood
   
            filename = f"code_final/score_results/iterations_ahc_gpt1_400_topk_0.25_diversity/step_{step}.csv"  
            with open(filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['step_id', 'smiles', 'valid', 'qed'])  
                scores = []
                diversity_scores = []
                smiles_counts = Counter(smiles)
                for s in smiles:
                    mol = Chem.MolFromSmiles(s)
                    is_valid = mol is not None  
                    qed_value = QED.qed(mol) if is_valid else 0  
                    scores.append(qed_value)
                    diversity_score = self._calculate_diversity_score(s, smiles,smiles_counts)
                    diversity_scores.append(diversity_score)
                    writer.writerow([step, s, is_valid, qed_value, diversity_score])  
            scores = torch.tensor(scores)
            diversity_scores = torch.tensor(diversity_scores)
            total_scores = scores+diversity_scores
          
            total_scores = torch.autograd.Variable(total_scores).cuda()
            loss = self._compute_loss_ahc(prior_likelihood, agent_likelihood, total_scores)
           
            loss = loss.mean()
            model.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            scaler.step(optimizer)
            scaler.update()
            self.tokens = 0
            if config.lr_decay:
                self.tokens += (seqs >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
                if self.tokens < config.warmup_tokens:
                    # linear warmup
                    lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                else:
                    # cosine learning rate decay
                    progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                    lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                lr = config.learning_rate * lr_mult
                for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
            else:
                lr = config.learning_rate
            
            if self.config.ckpt_path is not None :
                print(f'Saving the model')
                self.save_checkpoint()
            
        return None
        '''

    def train_reinvent(self, wandb):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)
        scaler = GradScaler()
        pattern =  "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)
        context = "C"
        x = torch.tensor([self.stoi[s] for s in regex.findall(context)], dtype=torch.long)[None,...].repeat(64, 1).to('cuda')
        for step in tqdm(range(400), total=400):
            seqs, probs, log_probs = sample(self.agent, x, 200,temperature=0.8, sample=True, top_k=10)
            seqs_x = torch.tensor(seqs[:,:-1], dtype=torch.long)
            seqs_y = torch.tensor(seqs[:,1:], dtype=torch.long)
            agent_likelihood,_,_,_ = self.agent(seqs_x,seqs_y)
            prior_likelihood,_,_,_ = self.prior(seqs_x,seqs_y)
            smiles = []
            valid_smiles = []
            for seq in seqs: 
                completion = ''.join([self.itos[int(i)] for i in seq])
                completion = completion.replace('<', '')
                smiles.append(completion)
                if Chem.MolFromSmiles(completion) is not None:  
                    valid_smiles.append(completion)
            agent_likelihood = - agent_likelihood
            print("The size of agent_likelihood:",agent_likelihood.size())
            prior_likelihoood = - prior_likelihood
            print("The size of prior_likelihood:",prior_likelihood.size())
            
            filename = f"NPDL-GEN&Transfer_learning/score_results/iterations_reinvent_gpt1_400/step_{step}.csv"
            with open(filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['step_id', 'smiles', 'valid', 'qed'])  
                scores = []
                for s in smiles:
                    mol = Chem.MolFromSmiles(s)
                    is_valid = mol is not None  
                    qed_value = QED.qed(mol) if is_valid else 0  
                    scores.append(qed_value)
                    writer.writerow([step, s, is_valid, qed_value])  
            scores = torch.tensor(scores)
            print("The size of scores:",scores.size())
            scores = torch.autograd.Variable(scores).cuda()
            loss = self._compute_loss_reinvent(prior_likelihood, agent_likelihood, scores)
           
            loss = loss.mean()
            model.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            scaler.step(optimizer)
            scaler.update()
            self.tokens = 0
            if config.lr_decay:
                self.tokens += (seqs >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
                if self.tokens < config.warmup_tokens:
                    # linear warmup
                    lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                else:
                    # cosine learning rate decay
                    progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                    lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                lr = config.learning_rate * lr_mult
                for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
            else:
                lr = config.learning_rate
            
            if self.config.ckpt_path is not None :
                print(f'Saving the model')
                self.save_checkpoint()
            
        return None
    
    
    def train_reinforce(self, wandb):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)
        scaler = GradScaler()
        pattern =  "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)
        context = "C"
        x = torch.tensor([self.stoi[s] for s in regex.findall(context)], dtype=torch.long)[None,...].repeat(64, 1).to('cuda')
        for step in tqdm(range(400), total=400):
            seqs, probs, log_probs = sample(self.agent, x, 200,temperature=0.8, sample=True, top_k=10)
            seqs_x = torch.tensor(seqs[:,:-1], dtype=torch.long)
            seqs_y = torch.tensor(seqs[:,1:], dtype=torch.long)
            agent_likelihood,_,_,_ = self.agent(seqs_x,seqs_y)
            smiles = []
            valid_smiles = []
            for seq in seqs: 
                completion = ''.join([self.itos[int(i)] for i in seq])
                completion = completion.replace('<', '')
                smiles.append(completion)
                if Chem.MolFromSmiles(completion) is not None:  
                    valid_smiles.append(completion)
            agent_likelihood = - agent_likelihood
            filename = f"NPDL-GEN&Transfer_learning/score_results/iterations_reinforce_gpt1_400/step_{step}.csv"
            with open(filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['step_id', 'smiles', 'valid', 'qed'])  
                scores = []
                for s in smiles:
                    mol = Chem.MolFromSmiles(s)
                    is_valid = mol is not None  
                    qed_value = QED.qed(mol) if is_valid else 0  
                    scores.append(qed_value)
                    writer.writerow([step, s, is_valid, qed_value])  
            scores = torch.tensor(scores)
            print("The size of scores:",scores.size())
            scores = torch.autograd.Variable(scores).cuda()
            loss = self._compute_loss_reinforce(agent_likelihood, scores)
           
            loss = loss.mean()
            model.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            scaler.step(optimizer)
            scaler.update()
            self.tokens = 0
            if config.lr_decay:
                self.tokens += (seqs >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
                if self.tokens < config.warmup_tokens:
                    # linear warmup
                    lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                else:
                    # cosine learning rate decay
                    progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                    lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                lr = config.learning_rate * lr_mult
                for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
            else:
                lr = config.learning_rate
            
            if self.config.ckpt_path is not None :
                print(f'Saving the model')
                self.save_checkpoint()
            
        return None
        


    def _compute_loss_reinvent(self, prior_likelihood, agent_likelihood, scores):
        sigma = 120
        augmented_likelihood = prior_likelihood + sigma * scores
        loss = torch.pow((augmented_likelihood - agent_likelihood), 2)
        self.record['loss'] += list(loss.detach().cpu().numpy())
        self.record['prior_nll'] += list(-prior_likelihood.detach().cpu().numpy())
        self.record['agent_nll'] += list(-agent_likelihood.detach().cpu().numpy())
        return loss
        
    def _compute_loss_ahc(self, prior_likelihood, agent_likelihood, scores):
        sigma = 120
        topk = 0.25
        augmented_likelihood = prior_likelihood + sigma * scores
        sscore, sscore_idxs = scores.sort(descending=True)
        loss = torch.pow((augmented_likelihood - agent_likelihood), 2)
        self.record['loss'] += list(loss.detach().cpu().numpy())
        self.record['prior_nll'] += list(-prior_likelihood.detach().cpu().numpy())
        self.record['agent_nll'] += list(-agent_likelihood.detach().cpu().numpy())
        loss = loss[sscore_idxs.data[:int(64 * topk)]]
        return loss
        
    def _calculate_diversity_score(self,smiles, all_smiles, smiles_counts):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0
        fp = Chem.RDKFingerprint(mol)
        similarities = []
        for s in all_smiles:
            other_mol = Chem.MolFromSmiles(s)
            if other_mol is not None:
                other_fp = Chem.RDKFingerprint(other_mol)
                similarity = Chem.DataStructs.FingerprintSimilarity(fp,other_fp)
                similarities.append(similarity)
        if similarities:
            diversity_score = 1-max(similarities)
        else:
            diversity_score = 1
        penalty = smiles_counts[smiles] - 1
        diversity_score -= penalty * 0.3
        return diversity_score
        
    def _compute_loss_reinforce(self, agent_likelihood, scores):  
        loss = agent_likelihood * scores
        self.record['loss'] += list(loss.detach().cpu().numpy())
        self.record['agent_nll'] += list(agent_likelihood.detach().cpu().numpy())
        return loss