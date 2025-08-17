# coding:latin-1
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def process_smiles_data(train_file, csv_files_pattern):
    with open(train_file, 'r') as f:
        train_smiles = set(f.read().splitlines())
    results = []
    for i in range(400):
        file_name = f'NPDL-GEN_Transfer_learning/draw_evaluate/score_results/iterations_ahc_gpt1_400_topk_0.25_diversity/step_{i}.csv'
        if not os.path.exists(file_name):
            print(f"ÎÄ¼þ {file_name} ²»´æÔÚ£¬Ìø¹ý¡£")
            continue
        
        df = pd.read_csv(file_name)
        valid_unique_smiles = set(df[df['valid'] == True]['smiles'])
        print(len(valid_unique_smiles))
        novel_smiles = valid_unique_smiles - train_smiles

        if len(valid_unique_smiles) > 0:
            ratio = len(novel_smiles) / len(valid_unique_smiles)
        else:
            ratio = 0
        
        results.append((file_name, ratio))
        print(f"´¦ÀíÍê³É {file_name}: ±ÈÀý = {ratio:.4f}")
    
    result_df = pd.DataFrame(results, columns=['File', 'Novel_Unique_Ratio'])
    result_df.to_csv('NPDL-GEN_Transfer_learning/draw_evaluate//smiles_analysis_results.csv', index=False)
    

train_file = 'NPDL-GEN_Transfer_learning/datasets/train_merged_smiles.txt'
csv_files_pattern = 'NPDL-GEN_Transfer_learning/draw_evaluate/score_results/iterations_ahc_gpt1_400_topk_0.25_diversity/step_{}.csv'
process_smiles_data(train_file, csv_files_pattern)


plt.rcParams['font.family'] = 'Times New Roman'
# ¶ÁÈ¡ smiles_analysis_results.csv ÎÄ¼þ
df = pd.read_csv('NPDL-GEN_Transfer_learning/draw_evaluate/smiles_analysis_results.csv')
# ´Ó File ÁÐÖÐÌáÈ¡ step ÊýÖµ
df['Step'] = df['File'].str.extract('step_(\d+)').astype(int)
# °´ step ÅÅÐò
df = df.sort_values(by='Step')
# »æÖÆ step Óë Novel_Unique_Ratio µÄÕÛÏßÍ¼
plt.figure(figsize=(7.7/2.54, 4.8/2.54))
plt.plot(df['Step'], df['Novel_Unique_Ratio'],linewidth=0.5)
plt.ylim(0,1.1)
plt.xlim(0, 400)
plt.tick_params(axis='both', which='major', labelsize=8) 
plt.xlabel('Epoch', fontsize=8)
plt.ylabel('Ratio of novel SMILES', fontsize=8)
plt.xticks([0, 50, 100, 150, 200, 250, 300, 350, 400], fontsize=8)
plt.yticks(fontsize=8)
plt.grid(False)
plt.savefig('NPDL-GEN_Transfer_learning/draw_evaluate/novelty_change_during_400_ahc_gpt1_0.25_diversity.svg', bbox_inches='tight',pad_inches=0)
