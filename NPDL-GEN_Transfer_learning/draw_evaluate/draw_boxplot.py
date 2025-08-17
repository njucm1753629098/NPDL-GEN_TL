import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

plt.rcParams['font.family'] = 'Times New Roman'
generation_df = pd.read_csv('NPDL-GEN_Transfer_learning/draw_evaluate/pretrain_gpt1_generation_10000_unique_with_property.csv')
inhibitor_df = pd.read_csv('NPDL-GEN_Transfer_learning/datasets/train_merged_smiles_with_property.csv')


generation_df_filtered = generation_df[(generation_df['mw'] < 1000) & (generation_df['logp'] > -5)].copy()
inhibitor_df_filtered = inhibitor_df[(inhibitor_df['mw'] < 1000) & (inhibitor_df['logp'] > -5)].copy()


generation_df_filtered.loc[:, 'source'] = 'gpt1 generation'
inhibitor_df_filtered.loc[:, 'source'] = 'training set'
combined_df = pd.concat([generation_df_filtered, inhibitor_df_filtered])

width_in_inches = 14 / 2.54 # 14.65
height_in_inches = 7 / 2.54
plt.figure(figsize=(width_in_inches,height_in_inches))
sns.set(style="whitegrid")
plt.rcParams['font.size'] = 8
properties = ['mw', 'logp', 'tpsa', 'qed', 'sas']
titles = ['Molecular Weight', 'LogP', 'TPSA', 'QED', 'SAS']
palette = {'training set':'#FCB373', 'gpt1 generation':'#7AADD2'}

fig, axes = plt.subplots(1,5, figsize=(width_in_inches, height_in_inches))
plt.subplots_adjust(wspace=0.1, left=0.05, right=0.99, bottom=0.15, top=0.9)

# Create subplots
for i, (prop, title, ax) in enumerate(zip(properties, titles, axes), 1):
   
    sns.boxplot(data=combined_df, x='source', y=prop, hue='source', palette=palette, width=0.7, showfliers=False, linewidth=1, ax=ax)
    ax.set_title(title, fontsize=8,pad=5)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.legend([],[], frameon=False) 
    ax.grid(False)
    '''
    # Adjust y-axis limits and ticks
    if prop == 'logp':
        plt.ylim(-1, 9)
        plt.yticks(range(0, 9, 2))
    elif prop == 'mw':
        plt.ylim(150, 650)
        plt.yticks(range(200, 651, 100))
    elif prop in ['hbd', 'hba']:
        plt.ylim(0, 11)
        plt.yticks(range(0, 11, 2))
    elif prop == 'rotbonds':
        plt.ylim(0, 15)
        plt.yticks(range(0, 15, 2))
    '''
    # Remove top and right spines
    sns.despine(ax=ax, top=True, right=True)
    
    # Adjust tick labels
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['training set', 'gpt1 generation'], rotation=45, ha='right', fontsize=8)
    

    ax.tick_params(axis='y', which='major', pad=0, labelsize=8)
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    #ax.set_facecolor('#FFF5F5')

# Adjust layout and save
plt.tight_layout()
plt.savefig('NPDL-GEN_Transfer_learning/draw_evaluate/molecular_properties_boxplot.svg', dpi=300, bbox_inches='tight')
plt.show()
