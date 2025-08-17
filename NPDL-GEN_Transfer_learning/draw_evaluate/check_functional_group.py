# coding:latin-1
 

'''
import csv
from rdkit import Chem
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# ��ȡ���ܻ� SMARTS �� SMILES ����
def read_data(smarts_file, smiles_file):
    with open(smarts_file, "r") as f:
        functional_groups = [line.strip() for line in f if line.strip()]
    
    with open(smiles_file, "r") as f:
        smiles_list = [line.strip() for line in f if line.strip()]
    
    return functional_groups, smiles_list

# ƥ�书�ܻ������ؽ��
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

# ���� CSV �ļ�
def save_csv(file_path, data, headers):
    with open(file_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(data)

# ������״ͼ
def plot_bar_chart(df):
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 8,  # ����ȫ�������СΪ8
        'axes.linewidth': 1.0,
        'axes.labelsize': 8,  # �������ǩ�����С
        'axes.titlesize': 8,  # ���������С
        'xtick.labelsize': 8,  # x��̶������С
        'ytick.labelsize': 8,  # y��̶������С
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
        'legend.fontsize': 8,  # ͼ�������С
        'legend.frameon': True,
        'legend.edgecolor': 'black'
    })

    # ����Ratio
    df['Ratio'] = df['Count'] / 3000

    # ����ͼ��
    fig, ax = plt.subplots(figsize=(3, 2.7), dpi=300)  # ����Ϊ���ʺ��ڿ��ĳߴ�

    # ������״ͼ
    bars = ax.bar(df['Name'], df['Ratio'],
                  color='#4472C4',     # ʹ�ø�רҵ����ɫ
                  edgecolor='black',
                  linewidth=0.8,
                  width=0.7)           # �������ӿ��

    # �����ֵ��ǩ
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom',
                fontsize=8)  # �����С����Ϊ8

    # �������ǩ
    ax.set_ylabel('Ratio', fontsize=8)

    # ����y�᷶Χ�͸�ʽ
    ax.set_ylim(0, 1.1)  # ����ֵ��ǩ�����ռ�
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # ����x���ǩ
    plt.xticks(rotation=45, ha='right')
    ax.set_xticklabels(df['Name'], fontsize=8)

    # ȥ��������
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)

    # �����߿�
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color('black')

    # ��������
    plt.tight_layout()

    # ����ͼƬ
    plt.savefig('autodl-tmp/code_final/functional_group_ratios_gpt1_second.svg',
                dpi=300,
                bbox_inches='tight',
                pad_inches=0)

    plt.close()

# ������
def main():
    functional_groups, smiles_list = read_data("autodl-tmp/code_final/functional.txt", "autodl-tmp/code_final/gpt1_ahc_3000.txt")
    
    # ƥ�书�ܻ���������
    output_rows, functional_group_count = match_functional_groups(smiles_list, functional_groups)
    save_csv("autodl-tmp/code_final/molecule_functional_gpt1.csv", output_rows, ["SMILES", "Functional Groups"])

    # ���湦���ų��ִ���
    stats_file = "autodl-tmp/code_final/functional_group_statistics_gpt1.csv"
    save_csv(stats_file, [[group, count] for group, count in functional_group_count.items()], ["Functional Group", "Count"])

    # ��ȡ������ͳ�����ݲ�����
    stats_df = pd.read_csv(stats_file)
    sorted_stats_df = stats_df.sort_values(by='Count', ascending=False).head(8)
    
    # ��ȡ���ܻ� Notes ӳ��
    functional_groups_df = pd.read_csv("autodl-tmp/code_final/FunctionalGroups.csv")
    smarts_to_notes = dict(zip(functional_groups_df['SMARTS'], functional_groups_df['Notes']))

    # ���� SMARTS ��ȡ Notes
    sorted_stats_df['Name'] = sorted_stats_df['Functional Group'].apply(lambda x: smarts_to_notes.get(x, 'Unknown'))

    # �������ս��
    sorted_stats_df.to_csv("autodl-tmp/code_final/sorted_functional_group_statistics_gpt1.csv", index=False)

    # ���Ʋ�������״ͼ
    plot_bar_chart(sorted_stats_df)

    # ��ӡͳ����Ϣ
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

# ��ȡ���ܻ� SMARTS �� SMILES ����
def read_data(smarts_file, smiles_file):
    with open(smarts_file, "r") as f:
        functional_groups = [line.strip() for line in f if line.strip()]
    
    with open(smiles_file, "r") as f:
        smiles_list = [line.strip() for line in f if line.strip()]
    
    return functional_groups, smiles_list

# ƥ�书�ܻ������ؽ��
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

# ���� CSV �ļ�
def save_csv(file_path, data, headers):
    with open(file_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(data)

# ������״ͼ
def plot_bar_chart(df):
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 8,  # ����ȫ�������СΪ8
        'axes.linewidth': 1.0,
        'axes.labelsize': 8,  # �������ǩ�����С
        'axes.titlesize': 8,  # ���������С
        'xtick.labelsize': 8,  # x��̶������С
        'ytick.labelsize': 8,  # y��̶������С
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
        'legend.fontsize': 8,  # ͼ�������С
        'legend.frameon': True,
        'legend.edgecolor': 'black'
    })

    # ����Ratio
    df['Ratio'] = df['Count'] / 3000

    # ����ͼ��
    fig, ax = plt.subplots(figsize=(3, 2.7), dpi=300)  # ����Ϊ���ʺ��ڿ��ĳߴ�

    # ������״ͼ
    bars = ax.bar(df['Name'], df['Ratio'],
                  color='#4472C4',     # ʹ�ø�רҵ����ɫ
                  edgecolor='black',
                  linewidth=0.8,
                  width=0.7)           # �������ӿ��

    # �����ֵ��ǩ
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom',
                fontsize=8)  # �����С����Ϊ8

    # �������ǩ
    ax.set_ylabel('Ratio', fontsize=8)

    # ����y�᷶Χ�͸�ʽ
    ax.set_ylim(0, 1.1)  # ����ֵ��ǩ�����ռ�
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # ����x���ǩ
    plt.xticks(rotation=45, ha='right')
    ax.set_xticklabels(df['Name'], fontsize=8)

    # ȥ��������
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)

    # �����߿�
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color('black')

    # ��������
    plt.tight_layout()

    # ����ͼƬ
    plt.savefig('autodl-tmp/code_final/functional_group_ratios_trainset_second.svg',
                dpi=300,
                bbox_inches='tight',
                pad_inches=0)

    plt.close()

# ������
def main():
    functional_groups, smiles_list = read_data("autodl-tmp/code_final/functional.txt", "autodl-tmp/code_final/datasets/zuhe/train_merged_smiles_with_property_scaffold_3000.txt")
    
    # ƥ�书�ܻ���������
    output_rows, functional_group_count = match_functional_groups(smiles_list, functional_groups)
    save_csv("autodl-tmp/code_final/molecule_functional_trainset.csv", output_rows, ["SMILES", "Functional Groups"])

    # ���湦���ų��ִ���
    stats_file = "autodl-tmp/code_final/functional_group_statistics_trainset.csv"
    save_csv(stats_file, [[group, count] for group, count in functional_group_count.items()], ["Functional Group", "Count"])

    # ��ȡ������ͳ�����ݲ�����
    stats_df = pd.read_csv(stats_file)
    sorted_stats_df = stats_df.sort_values(by='Count', ascending=False).head(8)
    
    # ��ȡ���ܻ� Notes ӳ��
    functional_groups_df = pd.read_csv("autodl-tmp/code_final/FunctionalGroups.csv")
    smarts_to_notes = dict(zip(functional_groups_df['SMARTS'], functional_groups_df['Notes']))

    # ���� SMARTS ��ȡ Notes
    sorted_stats_df['Name'] = sorted_stats_df['Functional Group'].apply(lambda x: smarts_to_notes.get(x, 'Unknown'))

    # �������ս��
    sorted_stats_df.to_csv("autodl-tmp/code_final/sorted_functional_group_statistics_trainset.csv", index=False)

    # ���Ʋ�������״ͼ
    plot_bar_chart(sorted_stats_df)

    # ��ӡͳ����Ϣ
    print(f"Total number of functional groups: {len(sorted_stats_df)}")
    print(f"Maximum ratio: {sorted_stats_df['Ratio'].max():.3f}")
    print(f"Average ratio: {sorted_stats_df['Ratio'].mean():.3f}")

if __name__ == "__main__":
    main()








