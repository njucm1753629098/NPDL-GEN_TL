# NPDL-GEN & Transfer Learning 
Our project aims to accelerate drug discovery by leveraging deep learning methods that include GPT1, AHC, and transfer learning to generate drug like natural products.The code in this repository is inspired on MolGPT ([https://github.com/devalab/molgpt](https://github.com/devalab/molgpt)) and AHC([https://github.com/MorganCThomas/SMILES-RNN](https://github.com/MorganCThomas/SMILES-RNN)).
## Requirements
- Python 3.8.10  
- rdkit 2022.9.5  
- wandb 0.16.3  
- pandas 1.5.3  
- torch 1.10.0+cu113  
- scikit-learn 1.3.0  
- numpy 1.21.4  
- tqdm 4.61.2  
- matplotlib 3.5.0  
- seaborn 0.11.0  
- json5 0.9.6  

Install all dependencies with:
```bash
pip install -r requirements.txt
```


## ## Usage
### 1. Pre-training
```bash
python NPDL-GEN_Transfer_learning/train/train.py \
  --run_name pretrain-8-layer-12 \
  --batch_size 64 \
  --max_epochs 8
```
**Note:** If encountering wandb issues, run `export WANDB_MODE=offline` first for offline training.
### 2. Generate Molecules from Pretrained Model
```bash
python NPDL-GEN_Transfer_learning/generate/generate.py \
  --model_weight weights/pretrain-8-layer-12.pt \
  --csv_name pretrain_generation_10000 \
  --gen_size 10000 \
  --batch_size 64
```
### 3. Reinforcement Learning Training
#### 3.1 Using QED Scoring
**NPDL-GEN:**
```bash
python NPDL-GEN_Transfer_learning/train/ahc.py \
  --run_name ahc-gpt1-400-topk-0.25 \
  --batch_size 64 \
  --max_epochs 400 \
  --model_weight weights/pretrain-8-layer-12.pt
```
**GPT1 + REINVENT:**
```bash
python NPDL-GEN_Transfer_learning/train/reinvent.py \
  --run_name reinvent-gpt1-400 \
  --batch_size 64 \
  --max_epochs 400 \
  --model_weight weights/pretrain-8-layer-12.pt
```
**GPT1 + REINFORCE:**
```bash
python NPDL-GEN_Transfer_learning/train/reinforce.py \
  --run_name reinforce-gpt1-400 \
  --batch_size 64 \
  --max_epochs 400 \
  --model_weight weights/pretrain-8-layer-12.pt
```
#### 3.2 Using QED + Diversity Scoring
```bash
python NPDL-GEN_Transfer_learning/train/ahc.py \
  --run_name ahc-gpt1-diversity-400-topk_0.25 \
  --batch_size 64 \
  --max_epochs 400 \
  --model_weight weights/pretrain-8-layer-12.pt
```
### 4. Generate Molecules from RL Model
```bash
python NPDL-GEN_Transfer_learning/train/ahc.py \
  --run_name ahc-gpt1-diversity-400-topk_0.25 \
  --batch_size 64 \
  --max_epochs 400 \
  --model_weight weights/pretrain-8-layer-12.pt
```
### 5. Transfer Learning Training
```bash
python NPDL-GEN_Transfer_learning/train/transfer_learning.py \
  --model_weight weights/pretrain-8-layer-12.pt \
  --run_name transfer-inflamnat-5 \
  --batch_size 64 \
  --max_epochs 10
```
### 6. Generate Molecules from Transfer Learning Model
```bash
python NPDL-GEN_Transfer_learning/generate/generate.py \
  --model_weight weights/transfer-inflamnat-5.pt \
  --csv_name transfer_generation_inflamnat \
  --gen_size 200 \
  --batch_size 64
```
### 7. Data Analysis, Visualization, etc.
Boxplot
```bash
python NPDL-GEN_Transfer_learning/code/draw_boxplot.py
```
QED Improvement Across Models
```bash
python NPDL-GEN_Transfer_learning/code/draw_qed_svg.py
```
QED vs Epoch
```bash
python NPDL-GEN_Transfer_learning/code/draw_qed_svg_after.py
```
Validity vs Epoch
```bash
python NPDL-GEN_Transfer_learning/code/draw_valid_change.py
```
Uniqueness vs Epoch
```bash
python NPDL-GEN_Transfer_learning/code/draw_unique_change.py
```
Novelty vs Epoch
```bash
python NPDL-GEN_Transfer_learning/code/draw_novelty_change.py
```
Violin Plot of QED
```bash
python NPDL-GEN_Transfer_learning/code/draw_qed_violin_plot.py
```
SAS Distribution
```bash
python NPDL-GEN_Transfer_learning/code/tongji_sas.py
```
NPClassifier Pathway Analysis
```bash
python NPDL-GEN_Transfer_learning/code/batch_npclassifier_pathway
```
molecula properties KDE Analysis
```bash
python NPDL-GEN_Transfer_learning/code/molecula_ properties_KDE.py
```

## Notes

-   Replace placeholder paths (e.g., `weights/...`) with actual file paths