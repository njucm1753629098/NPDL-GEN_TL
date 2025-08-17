import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import matplotlib.patches as patches

if __name__ == '__main__':
    generation_df = pd.read_csv('code_final/datasets/transfer_generation_inflamnat_with_property.csv')
    inhibitor_df = pd.read_csv('code_final/datasets/inflamnat_after_filter_with_property.csv')
    sns.set(style="whitegrid")
    
    font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'
    prop = fm.FontProperties(fname=font_path, size=18, weight='bold')
    legend_prop = fm.FontProperties(fname=font_path, size=16, weight='bold')
    
    columns = ['mw', 'logp', 'tpsa', 'hbd', 'hba', 'rotbonds']
    titles = ['Molecular Weight', 'LogP', 'TPSA', 'HBD', 'HBA', 'Number of Rotatable Bonds']
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for idx, column in enumerate(columns):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        color_hist_fine = '#8856a7'
        color_hist_gen = '#43a2ca'
        color_kde_fine = '#e34a33'
        color_kde_gen = '#2c7fb8'
        # color='#CC99CC'color='#B5CDE1'color='red'color='black'
        sns.histplot(data=inhibitor_df, x=column, kde=False, color=color_hist_fine, label='hist: fine-tuned data', stat='density', bins=30, alpha=1, ax=ax)
        sns.histplot(data=generation_df, x=column, kde=False, color=color_hist_gen, label='hist: generated data', stat='density', bins=30, alpha=1, ax=ax)
        sns.kdeplot(data=inhibitor_df[column], color=color_kde_fine, label='KDE: fine-tuned data', linestyle='-', linewidth=2, ax=ax)
        sns.kdeplot(data=generation_df[column], color=color_kde_gen, label='KDE: generated data', linestyle='--', linewidth=2, ax=ax)
        
        ax.set_title(titles[idx], fontsize=18, fontproperties=prop)
        if col == 0:
            ax.set_ylabel('Density', fontsize=18, fontproperties=prop)
        else:
            ax.set_ylabel('')
        
        ax.set_xlabel('')
        
        # Set specific ranges for mw, tpsa, and rotbonds
        if column == 'mw':
            ax.set_xlim(0, 600)
        elif column == 'tpsa':
            ax.set_xlim(0, 150)
        elif column == 'rotbonds':
            ax.set_xlim(0, 6)
        
        ax.grid(False)
    
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', prop=legend_prop, fontsize=16, ncol=4, frameon=False)
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig(f'code_final/kde_density_plots.svg')
    plt.show()