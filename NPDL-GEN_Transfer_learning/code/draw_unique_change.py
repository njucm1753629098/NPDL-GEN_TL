import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.family'] = 'Times New Roman'
unique_ratios = []

for step in range(400):
    #filename = f"autodl-tmp/code_final/score_results/iterations_ahc_gpt1_400_topk_0.25/step_{step}.csv"
    filename = f"autodl-tmp/code_final/score_results/iterations_ahc_gpt1_400_topk_0.25_diversity/step_{step}.csv"
    df = pd.read_csv(filename)
    valid_smiles = df[df['valid'] == True]['smiles']
    if len(valid_smiles) == 0:
        unique_ratio = 0
    else:
        unique_ratio = valid_smiles.nunique() / len(valid_smiles)
    unique_ratios.append(unique_ratio)


steps = list(range(400))
plt.figure(figsize=(7.7/2.54, 4.8/2.54))
plt.plot(steps, unique_ratios,linewidth=0.5)
plt.xlabel('Epoch', fontsize=8)  
plt.ylabel('Ratio of unique SMILES', fontsize=8)  
 

plt.ylim(0, 1.1)
plt.xlim(0, 400)
plt.tick_params(axis='both', which='major', labelsize=8)  

plt.grid(False)
plt.xticks([0, 50, 100, 150, 200, 250, 300, 350, 400], fontsize=8)
plt.yticks(fontsize=8)
#plt.savefig("autodl-tmp/code_final/unique_change_during_400_ahc_gpt1_0.25.svg", dpi=300, bbox_inches='tight', pad_inches=0)
plt.savefig("autodl-tmp/code_final/unique_change_during_400_ahc_gpt1_0.25_diversity.svg", dpi=300, bbox_inches='tight', pad_inches=0)
plt.show()


