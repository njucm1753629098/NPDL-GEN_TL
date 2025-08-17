# coding:latin-1
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors

def plot_stereoisomers(csv_file):
    # ��ȡ�ʹ�������
    df = pd.read_csv(csv_file)
    df['mol_weight'] = df.apply(lambda x: Descriptors.ExactMolWt(Chem.MolFromSmiles(x['SMILES_1'])), axis=1)
    
    bins = [100, 150, 200, 250, 300, 350, 400, 450, 500]
    labels = ['100-150', '150-200', '200-250', '250-300', '300-350', '350-400', '400-450', '450-500']
    df = df[df['mol_weight'].between(100, 500)]
    df['weight_range'] = pd.cut(df['mol_weight'], bins=bins, labels=labels)
    
    # ͳ������
    result = pd.pivot_table(
        df,
        values='SMILES_1',
        index='weight_range',
        columns='Type',
        aggfunc='count',
        fill_value=0
    )
    
    # ��ͼ����
    plt.figure(figsize=(4.5, 2.2))
    x = range(len(result.index))
    width = 0.2

    # ��������Ϊ Times New Roman
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 8,  # �Ŵ������С
        'axes.titlesize': 8,
        'axes.labelsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8
    })
    
    # ������״ͼ
    bar1 = plt.bar(x, result['Enantiomer'] if 'Enantiomer' in result.columns else [0]*len(x), 
                   width, label='Enantiomers', color='skyblue')
    bar2 = plt.bar([i + width for i in x], result['Diastereomer'] if 'Diastereomer' in result.columns else [0]*len(x), 
                   width, label='Diastereomers', color='lightcoral')
    
    # ���ñ�ǩ�ͱ���
    plt.xlabel('Molecular Weight')
    plt.ylabel('Number of Pairs')
    plt.title('Distribution of Enantiomers/Diastereomers by Molecular Weight')
    plt.xticks([i + width/2 for i in x], labels)  # �������ǩ����ת
    plt.legend()
    
    # �������Ϸ���ע����
    max_height = 0 #���ڸ������ӵ����߶�
    for bar_group in [bar1, bar2]:
        for bar in bar_group:
            height = bar.get_height()
            max_height = max(max_height, height)
            if height > 0:  # ֻ��ע����ֵ
                plt.text(bar.get_x() + bar.get_width() / 2, height + 0.5, f'{int(height)}', 
                         ha='center', va='bottom', fontsize=8)
    plt.ylim(0, max_height + 9)
    # ��������
    plt.tight_layout()

    # �������ʾ
    plt.savefig('autodl-tmp/code_final/stereoisomers_distribution_gpt1.svg', dpi=300)
    #plt.savefig('autodl-tmp/code_final/stereoisomers_distribution_trainset.svg', dpi=300)
    plt.show()

# ���ú�������ͼ��
plot_stereoisomers("autodl-tmp/code_final/yigouti_gpt1.csv")
#plot_stereoisomers("autodl-tmp/code_final/yigouti_trainset.csv")


 
