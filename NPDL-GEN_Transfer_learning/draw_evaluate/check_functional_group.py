# coding:latin-1
 

'''
import csv
from rdkit import Chem
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# ¶ÁÈ¡¹¦ÄÜ»ù SMARTS ºÍ SMILES Êý¾Ý
def read_data(smarts_file, smiles_file):
    with open(smarts_file, "r") as f:
        functional_groups = [line.strip() for line in f if line.strip()]
    
    with open(smiles_file, "r") as f:
        smiles_list = [line.strip() for line in f if line.strip()]
    
    return functional_groups, smiles_list

# Æ¥Åä¹¦ÄÜ»ù²¢·µ»Ø½á¹û
def match_functional_groups(smiles_list, functional_groups):
    output_rows = []
    functional_group_count = Counter()
    
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            matched_groups = [smarts for smarts in functional_groups if Chem.MolFromSmarts(smarts) and mol.HasSubstructMatch(Chem.MolFromSmarts(smarts))]
            for group in matched_groups:
                functional_group_count[group] += 1
            output_rows.append([smiles, ", ".join(matched_groups) if matched_groups else "None"])
        else:
            output_rows.append([smiles, "Invalid SMILES"])
    
    return output_rows, functional_group_count

# ±£´æ CSV ÎÄ¼þ
def save_csv(file_path, data, headers):
    with open(file_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(data)

# »æÖÆÖù×´Í¼
def plot_bar_chart(df):
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 8,  # ÉèÖÃÈ«¾Ö×ÖÌå´óÐ¡Îª8
        'axes.linewidth': 1.0,
        'axes.labelsize': 8,  # ×ø±êÖá±êÇ©×ÖÌå´óÐ¡
        'axes.titlesize': 8,  # ±êÌâ×ÖÌå´óÐ¡
        'xtick.labelsize': 8,  # xÖá¿Ì¶È×ÖÌå´óÐ¡
        'ytick.labelsize': 8,  # yÖá¿Ì¶È×ÖÌå´óÐ¡
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
        'legend.fontsize': 8,  # Í¼Àý×ÖÌå´óÐ¡
        'legend.frameon': True,
        'legend.edgecolor': 'black'
    })

    # ¼ÆËãRatio
    df['Ratio'] = df['Count'] / 3000

    # ´´½¨Í¼ÐÎ
    fig, ax = plt.subplots(figsize=(3, 2.7), dpi=300)  # µ÷ÕûÎª¸üÊÊºÏÆÚ¿¯µÄ³ß´ç

    # »æÖÆÖù×´Í¼
    bars = ax.bar(df['Name'], df['Ratio'],
                  color='#4472C4',     # Ê¹ÓÃ¸ü×¨ÒµµÄÀ¶É«
                  edgecolor='black',
                  linewidth=0.8,
                  width=0.7)           # µ÷ÕûÖù×Ó¿í¶È

    # Ìí¼ÓÊýÖµ±êÇ©
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom',
                fontsize=8)  # ×ÖÌå´óÐ¡µ÷ÕûÎª8

    # ÉèÖÃÖá±êÇ©
    ax.set_ylabel('Ratio', fontsize=8)

    # ÉèÖÃyÖá·¶Î§ºÍ¸ñÊ½
    ax.set_ylim(0, 1.1)  # ¸øÊýÖµ±êÇ©Áô³ö¿Õ¼ä
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # µ÷ÕûxÖá±êÇ©
    plt.xticks(rotation=45, ha='right')
    ax.set_xticklabels(df['Name'], fontsize=8)

    # È¥³ýÍø¸ñÏß
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)

    # µ÷Õû±ß¿ò
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color('black')

    # µ÷Õû²¼¾Ö
    plt.tight_layout()

    # ±£´æÍ¼Æ¬
    plt.savefig('autodl-tmp/code_final/functional_group_ratios_gpt1_second.svg',
                dpi=300,
                bbox_inches='tight',
                pad_inches=0)

    plt.close()

# Ö÷º¯Êý
def main():
    functional_groups, smiles_list = read_data("autodl-tmp/code_final/functional.txt", "autodl-tmp/code_final/gpt1_ahc_3000.txt")
    
    # Æ¥Åä¹¦ÄÜ»ù²¢±£´æ½á¹û
    output_rows, functional_group_count = match_functional_groups(smiles_list, functional_groups)
    save_csv("autodl-tmp/code_final/molecule_functional_gpt1.csv", output_rows, ["SMILES", "Functional Groups"])

    # ±£´æ¹¦ÄÜÍÅ³öÏÖ´ÎÊý
    stats_file = "autodl-tmp/code_final/functional_group_statistics_gpt1.csv"
    save_csv(stats_file, [[group, count] for group, count in functional_group_count.items()], ["Functional Group", "Count"])

    # ¶ÁÈ¡¹¦ÄÜÍÅÍ³¼ÆÊý¾Ý²¢ÅÅÐò
    stats_df = pd.read_csv(stats_file)
    sorted_stats_df = stats_df.sort_values(by='Count', ascending=False).head(8)
    
    # ¶ÁÈ¡¹¦ÄÜ»ù Notes Ó³Éä
    functional_groups_df = pd.read_csv("autodl-tmp/code_final/FunctionalGroups.csv")
    smarts_to_notes = dict(zip(functional_groups_df['SMARTS'], functional_groups_df['Notes']))

    # ¸ù¾Ý SMARTS »ñÈ¡ Notes
    sorted_stats_df['Name'] = sorted_stats_df['Functional Group'].apply(lambda x: smarts_to_notes.get(x, 'Unknown'))

    # ±£´æ×îÖÕ½á¹û
    sorted_stats_df.to_csv("autodl-tmp/code_final/sorted_functional_group_statistics_gpt1.csv", index=False)

    # »æÖÆ²¢±£´æÖù×´Í¼
    plot_bar_chart(sorted_stats_df)

    # ´òÓ¡Í³¼ÆÐÅÏ¢
    print(f"Total number of functional groups: {len(sorted_stats_df)}")
    print(f"Maximum ratio: {sorted_stats_df['Ratio'].max():.3f}")
    print(f"Average ratio: {sorted_stats_df['Ratio'].mean():.3f}")

if __name__ == "__main__":
    main()

'''
import csv
from rdkit import Chem
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# ¶ÁÈ¡¹¦ÄÜ»ù SMARTS ºÍ SMILES Êý¾Ý
def read_data(smarts_file, smiles_file):
    with open(smarts_file, "r") as f:
        functional_groups = [line.strip() for line in f if line.strip()]
    
    with open(smiles_file, "r") as f:
        smiles_list = [line.strip() for line in f if line.strip()]
    
    return functional_groups, smiles_list

# Æ¥Åä¹¦ÄÜ»ù²¢·µ»Ø½á¹û
def match_functional_groups(smiles_list, functional_groups):
    output_rows = []
    functional_group_count = Counter()
    
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            matched_groups = [smarts for smarts in functional_groups if Chem.MolFromSmarts(smarts) and mol.HasSubstructMatch(Chem.MolFromSmarts(smarts))]
            for group in matched_groups:
                functional_group_count[group] += 1
            output_rows.append([smiles, ", ".join(matched_groups) if matched_groups else "None"])
        else:
            output_rows.append([smiles, "Invalid SMILES"])
    
    return output_rows, functional_group_count

# ±£´æ CSV ÎÄ¼þ
def save_csv(file_path, data, headers):
    with open(file_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(data)

# »æÖÆÖù×´Í¼
def plot_bar_chart(df):
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 8,  # ÉèÖÃÈ«¾Ö×ÖÌå´óÐ¡Îª8
        'axes.linewidth': 1.0,
        'axes.labelsize': 8,  # ×ø±êÖá±êÇ©×ÖÌå´óÐ¡
        'axes.titlesize': 8,  # ±êÌâ×ÖÌå´óÐ¡
        'xtick.labelsize': 8,  # xÖá¿Ì¶È×ÖÌå´óÐ¡
        'ytick.labelsize': 8,  # yÖá¿Ì¶È×ÖÌå´óÐ¡
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
        'legend.fontsize': 8,  # Í¼Àý×ÖÌå´óÐ¡
        'legend.frameon': True,
        'legend.edgecolor': 'black'
    })

    # ¼ÆËãRatio
    df['Ratio'] = df['Count'] / 3000

    # ´´½¨Í¼ÐÎ
    fig, ax = plt.subplots(figsize=(3, 2.7), dpi=300)  # µ÷ÕûÎª¸üÊÊºÏÆÚ¿¯µÄ³ß´ç

    # »æÖÆÖù×´Í¼
    bars = ax.bar(df['Name'], df['Ratio'],
                  color='#4472C4',     # Ê¹ÓÃ¸ü×¨ÒµµÄÀ¶É«
                  edgecolor='black',
                  linewidth=0.8,
                  width=0.7)           # µ÷ÕûÖù×Ó¿í¶È

    # Ìí¼ÓÊýÖµ±êÇ©
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom',
                fontsize=8)  # ×ÖÌå´óÐ¡µ÷ÕûÎª8

    # ÉèÖÃÖá±êÇ©
    ax.set_ylabel('Ratio', fontsize=8)

    # ÉèÖÃyÖá·¶Î§ºÍ¸ñÊ½
    ax.set_ylim(0, 1.1)  # ¸øÊýÖµ±êÇ©Áô³ö¿Õ¼ä
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # µ÷ÕûxÖá±êÇ©
    plt.xticks(rotation=45, ha='right')
    ax.set_xticklabels(df['Name'], fontsize=8)

    # È¥³ýÍø¸ñÏß
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)

    # µ÷Õû±ß¿ò
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color('black')

    # µ÷Õû²¼¾Ö
    plt.tight_layout()

    # ±£´æÍ¼Æ¬
    plt.savefig('autodl-tmp/code_final/functional_group_ratios_trainset_second.svg',
                dpi=300,
                bbox_inches='tight',
                pad_inches=0)

    plt.close()

# Ö÷º¯Êý
def main():
    functional_groups, smiles_list = read_data("autodl-tmp/code_final/functional.txt", "autodl-tmp/code_final/datasets/zuhe/train_merged_smiles_with_property_scaffold_3000.txt")
    
    # Æ¥Åä¹¦ÄÜ»ù²¢±£´æ½á¹û
    output_rows, functional_group_count = match_functional_groups(smiles_list, functional_groups)
    save_csv("autodl-tmp/code_final/molecule_functional_trainset.csv", output_rows, ["SMILES", "Functional Groups"])

    # ±£´æ¹¦ÄÜÍÅ³öÏÖ´ÎÊý
    stats_file = "autodl-tmp/code_final/functional_group_statistics_trainset.csv"
    save_csv(stats_file, [[group, count] for group, count in functional_group_count.items()], ["Functional Group", "Count"])

    # ¶ÁÈ¡¹¦ÄÜÍÅÍ³¼ÆÊý¾Ý²¢ÅÅÐò
    stats_df = pd.read_csv(stats_file)
    sorted_stats_df = stats_df.sort_values(by='Count', ascending=False).head(8)
    
    # ¶ÁÈ¡¹¦ÄÜ»ù Notes Ó³Éä
    functional_groups_df = pd.read_csv("autodl-tmp/code_final/FunctionalGroups.csv")
    smarts_to_notes = dict(zip(functional_groups_df['SMARTS'], functional_groups_df['Notes']))

    # ¸ù¾Ý SMARTS »ñÈ¡ Notes
    sorted_stats_df['Name'] = sorted_stats_df['Functional Group'].apply(lambda x: smarts_to_notes.get(x, 'Unknown'))

    # ±£´æ×îÖÕ½á¹û
    sorted_stats_df.to_csv("autodl-tmp/code_final/sorted_functional_group_statistics_trainset.csv", index=False)

    # »æÖÆ²¢±£´æÖù×´Í¼
    plot_bar_chart(sorted_stats_df)

    # ´òÓ¡Í³¼ÆÐÅÏ¢
    print(f"Total number of functional groups: {len(sorted_stats_df)}")
    print(f"Maximum ratio: {sorted_stats_df['Ratio'].max():.3f}")
    print(f"Average ratio: {sorted_stats_df['Ratio'].mean():.3f}")

if __name__ == "__main__":
    main()








