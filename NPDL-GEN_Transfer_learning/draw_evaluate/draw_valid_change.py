import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
plt.rcParams['font.family'] = 'Times New Roman'
 


valid_ratios = []


for step in range(400):
    filename = f"NPDL-GEN_Transfer_learning/draw_evaluate/score_results/iterations_ahc_gpt1_400_topk_0.25_diversity/step_{step}.csv"
    df = pd.read_csv(filename)
    valid_ratio = df['valid'].mean()  
    valid_ratios.append(valid_ratio)


       
steps = list(range(400))


plt.figure(figsize=(7.7/2.54, 4.8/2.54))
plt.plot(steps, valid_ratios,linewidth=0.5)
plt.xlabel('Epoch', fontsize=8)  
plt.ylabel('Ratio of valid SMILES', fontsize=8)  
#plt.title('Valid Ratio vs. Step', fontsize=20)  
plt.ylim(0, 1.1)  
plt.xlim(0, 400)
plt.tick_params(axis='both', which='major', labelsize=8)
plt.grid(False)
plt.xticks([0, 50, 100, 150, 200, 250, 300, 350, 400], fontsize=8)
plt.yticks(fontsize=8)
plt.savefig("NPDL-GEN_Transfer_learning/draw_evaluate/valid_change_during_400_ahc_gpt1_0.25_diversity.svg", dpi=300, bbox_inches='tight',pad_inches=0)

