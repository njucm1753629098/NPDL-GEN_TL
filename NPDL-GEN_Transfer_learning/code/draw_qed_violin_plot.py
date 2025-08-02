# coding:latin-1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd
from scipy import stats
from tabulate import tabulate



import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.font_manager as fm
plt.style.use('seaborn')

# 1) ¾ø¶ÔÂ·¾¶£¨°´Äã»úÆ÷Êµ¼ÊÎÄ¼þÌîÐ´£©
times_path = '/usr/share/fonts/truetype/msttcorefonts/times.ttf'

# 2) °Ñ×ÖÌåÎÄ¼þ×¢²áµ½ matplotlib£¬²¢È¡Ò»¸öÄÚ²¿Ãû×Ö
fm.fontManager.addfont(times_path)          # Ö»ÔÚ matplotlib ¡Ý 3.2 ÓÐÐ§
rcParams['font.family'] = 'Times New Roman'  # ÏÖÔÚÕâ¸öÃû×ÖÒ»¶¨´æÔÚ

def get_qed_from_file(filename, n_samples=1800):
    if filename.endswith('.csv'):
        df = pd.read_csv(filename)
        smiles = df['SMILES'].sample(n=n_samples, random_state=42).tolist()
    else:  
        with open(filename, 'r') as f:
            smiles = f.readlines()
        smiles = [s.strip() for s in smiles]
        smiles = np.random.choice(smiles, n_samples, replace=False)
    
    qed_values = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            qed_values.append(Descriptors.qed(mol))
    
    return qed_values


data = {
    'MolGPT': get_qed_from_file('code_final/gpt_ahc_diversity_generation_10000_valid_unique.csv'),
    'VAE': get_qed_from_file('code_final/vae_ahc2_early_diversity_generation_10000_valid_unique.csv'),
    'CharRNN': get_qed_from_file('code_final/char_rnn2_ahc_early_diversity_generation_10000_valid_unique.csv'),
    'NP-GEN': get_qed_from_file('code_final/gpt1_ahc_400_topk_0.25_diversity_generation_10000.csv')
}

# Calculate the proportion of molecules with QED < 0.2 for each model
qed_threshold = 0.2
model_low_qed_ratio = {}

for model, qed_list in data.items():
    if len(qed_list) == 0:
        ratio = 0
    else:
        low_qed_count = sum(q < qed_threshold for q in qed_list)
        ratio = low_qed_count / len(qed_list)
    model_low_qed_ratio[model] = ratio

# Print out the results
for model, ratio in model_low_qed_ratio.items():
    print(f"{model}: {ratio:.3f}")


'''
def display_statistics(statistics):
    # ×¼±¸±í¸ñÊý¾Ý
    headers = ['Model', 'Mean', 'SD', 'Median']
    table_data = []
    for model, stats in statistics.items():
        row = [model, stats['mean'], stats['sd'], stats['median']]
        table_data.append(row)
    
    # Ê¹ÓÃtabulate´òÓ¡¸ñÊ½»¯±í¸ñ
    print(tabulate(table_data, headers=headers, tablefmt='grid'))


def calculate_statistics(data):
    results = {}
    for key, values in data.items():
        mean = np.mean(values)
        sd = np.std(values)
        median = np.median(values)
        results[key] = {
            'mean': mean,
            'sd': sd,
            'median': median
        }
    return results

statistics = calculate_statistics(data)
display_statistics(statistics)

def save_to_csv(statistics, filename='model_statistics.csv'):
    df = pd.DataFrame.from_dict(statistics, orient='index')
    df.to_csv(filename)
    print(f"\nStatistics saved to {filename}")

save_to_csv(statistics)

'''





plt.figure(figsize=(12/2.54, 7/2.54))


violin_parts = plt.violinplot([data[key] for key in ['CharRNN', 'MolGPT', 'VAE', 'NP-GEN']], 
                              showmeans=False, showmedians=False, showextrema=False)


colors = ['blue', 'orange', 'green', 'red']
for i, pc in enumerate(violin_parts['bodies']):
    pc.set_facecolor(colors[i])
    pc.set_edgecolor('black')
    pc.set_alpha(1)

positions = range(1, 5)
for i, d in enumerate([data[key] for key in ['CharRNN', 'MolGPT', 'VAE', 'NP-GEN']]):
    quantiles = np.percentile(d, [25, 50, 75])
    plt.vlines(positions[i], quantiles[0], quantiles[2], color='k', linestyle='-', lw=2)
    plt.vlines(positions[i], quantiles[1], quantiles[1], color='w', linestyle='-', lw=2)


plt.ylim(0, 1)
plt.ylabel('QED', fontsize=10)
plt.xticks(positions, ['CharRNN+AHC', 'MolGPT+AHC', 'VAE+AHC', 'NPDL-GEN'], fontsize=10)
plt.yticks(fontsize=10)


sns.despine()


plt.tight_layout()
plt.savefig('code_final/qed_violin_plot_414.svg', dpi=300, bbox_inches='tight', pad_inches=0)
plt.show()