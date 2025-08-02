# coding:latin-1
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random








import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.font_manager as fm


# 1) ¾ø¶ÔÂ·¾¶£¨°´Äã»úÆ÷Êµ¼ÊÎÄ¼þÌîÐ´£©
times_path = '/usr/share/fonts/truetype/msttcorefonts/times.ttf'

# 2) °Ñ×ÖÌåÎÄ¼þ×¢²áµ½ matplotlib£¬²¢È¡Ò»¸öÄÚ²¿Ãû×Ö
fm.fontManager.addfont(times_path)          # Ö»ÔÚ matplotlib ¡Ý 3.2 ÓÐÐ§
rcParams['font.family'] = 'Times New Roman'  # ÏÖÔÚÕâ¸öÃû×ÖÒ»¶¨´æÔÚ














#plt.rcParams['font.family'] = 'Times New Roman'


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


reinvent_means = calculate_qed_mean('code_final/score_results/iterations_reinvent_gpt1_400')
ahc_means = calculate_qed_mean('code_final/score_results/iterations_ahc_gpt1_400')
reinforce_means = calculate_qed_mean('code_final/score_results/iterations_reinforce_gpt1_400')


plt.figure(figsize=(7.7/2.54, 4.8/2.54))


plt.plot(range(400), reinvent_means, label='REINVENT', color='blue', linewidth=0.5)
plt.plot(range(400), ahc_means, label='AHC', color='green', linewidth=0.5)
plt.plot(range(400), reinforce_means, label='REINFORCE', color='red', linewidth=0.5)


plt.xlabel('Epoch', fontsize=8)
plt.ylabel('Average of qed values', fontsize=8)
plt.xticks([0, 50, 100, 150, 200, 250, 300, 350, 400],fontsize=8)
plt.yticks(fontsize=8)

plt.legend(loc='lower right', fontsize=8)


plt.ylim(0, 1)
plt.xlim(0,400)
plt.grid(False)
plt.savefig('code_final/ahc_reinvent_reinforce_qed_compare.svg',dpi=300, bbox_inches='tight', pad_inches=0)
plt.show()



'''
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


reinvent_means = calculate_qed_mean('autodl-tmp/MolGPT/score_results/iterations_ahc_gpt1_400_topk_0.25_diversity') # iterations_ahc_gpt1_400_topk_0.25_qs


# ±£´æÁÐ±íµ½ÎÄ±¾ÎÄ¼þ
output_path = 'autodl-tmp/MolGPT/result/qed_after.txt'

with open(output_path, 'w') as f:
    for value in reinvent_means:
        if value is not None:
            f.write(f"{value:.6f}\n")  # ±£Áô6Î»Ð¡Êý£¬¿É°´ÐèÐÞ¸Ä
        else:
            f.write("None\n")
'''

'''
plt.figure(figsize=(7.7/2.54, 4.8/2.54))


plt.plot(range(400), reinvent_means,  linewidth=0.5)


plt.xlabel('Epoch', fontsize=8)
plt.ylabel('Average of qed values', fontsize=8)
plt.xticks([0, 50, 100, 150, 200, 250, 300, 350, 400], fontsize=8)
plt.yticks(fontsize=8)



plt.ylim(0, 1)
plt.xlim(0, 400)


plt.grid(False)


#plt.savefig('autodl-tmp/MolGPT/qed_change_during_400_ahc_gpt1_topk_0.25_qs.svg', dpi=300, bbox_inches='tight', pad_inches=0)
plt.savefig('autodl-tmp/MolGPT/qed_change_during_400_ahc_gpt1_topk_0.25_diversity.svg', dpi=300, bbox_inches='tight', pad_inches=0)


plt.show()
'''