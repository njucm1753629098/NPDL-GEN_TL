import os
import pandas as pd
import matplotlib.pyplot as plt

def calculate_qed_mean(directory):
    qed_means = []
    
    
    for i in range(400):
        file_name = f'step_{i}.csv'
        file_path = os.path.join(directory, file_name)
        
        
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            if 'qed' in df.columns:
                qed_mean = df['qed'].mean()
                qed_means.append(qed_mean)
            else:
                qed_means.append(None)  
        else:
            qed_means.append(None)  
    
    return qed_means


qed_means = calculate_qed_mean('NPDL-GEN_Transfer_learning/draw_evaluate/score_results/iterations_ahc_gpt1_400_topk_0.25_diversity') 
plt.figure(figsize=(7.7/2.54, 4.8/2.54))
plt.plot(range(400), qed_means,  linewidth=0.5)
plt.xlabel('Epoch', fontsize=8)
plt.ylabel('Average of qed values', fontsize=8)
plt.xticks([0, 50, 100, 150, 200, 250, 300, 350, 400], fontsize=8)
plt.yticks(fontsize=8)
plt.ylim(0, 1)
plt.xlim(0, 400)
plt.grid(False)
#plt.savefig('autodl-tmp/MolGPT/qed_change_during_400_ahc_gpt1_topk_0.25_qs.svg', dpi=300, bbox_inches='tight', pad_inches=0)
plt.savefig('NPDL-GEN_Transfer_learning/draw_evaluate/qed_change_during_400_ahc_gpt1_topk_0.25_diversity.svg', dpi=300, bbox_inches='tight', pad_inches=0)

