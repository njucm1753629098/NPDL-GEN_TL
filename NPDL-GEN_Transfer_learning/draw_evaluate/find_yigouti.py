# coding:latin-1
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from itertools import combinations
import pandas as pd

def is_enantiomer(mol1, mol2):
    """�ж��������Ƿ�Ϊ��ӳ�칹��"""
    
    # �ж������ӵķ���ʽ�Ƿ���ͬ
    formula1 = rdMolDescriptors.CalcMolFormula(mol1)
    formula2 = rdMolDescriptors.CalcMolFormula(mol2)
    
    if formula1 != formula2:
        return False # �������ʽ��ͬ���Ͳ��Ƕ�ӳ�칹��
    
    # ������������
    conf1 = Chem.FindMolChiralCenters(mol1, includeUnassigned=True) 
    # ����������������������λ�ú�(����)�����Ա�־������ 1 ���������ģ�[(1, 'S')]������ 2 ���������ģ�[(1, 'R')]�����ڶ�ӳ�칹�壬���Ա�־ 'S' �� 'R' ���෴�ģ���˷�����������ӳ�칹���������ͬ��λ���Ͼ�����������
    conf2 = Chem.FindMolChiralCenters(mol2, includeUnassigned=True)
    
    # �ж��������Ƿ�����������
    if not conf1 or not conf2:
        return False  # ��������κ�һ������û���������ģ����Ƕ�ӳ�칹��
    
    # �ж��������������������Ƿ���ͬ
    if len(conf1) != len(conf2):
        return False  # ��������������ͬ���϶����Ƕ�ӳ�칹��
    
    # ������������Ƿ���ȫ�෴
    for (idx1, chiral1), (idx2, chiral2) in zip(conf1, conf2):
        if idx1 != idx2 or chiral1 == chiral2:  # ����������ͬ��λ�ò�ƥ��
            return False
    return True

def is_diastereomer(mol1, mol2):
    """�ж��������Ƿ�Ϊ�Ƕ�ӳ�칹��"""
    
    # �ж������ӵķ���ʽ�Ƿ���ͬ
    formula1 = rdMolDescriptors.CalcMolFormula(mol1)
    formula2 = rdMolDescriptors.CalcMolFormula(mol2)
    
    if formula1 != formula2:
        return False # �������ʽ��ͬ���Ͳ��ǷǶ�ӳ�칹��
        
    conf1 = Chem.FindMolChiralCenters(mol1, includeUnassigned=True)
    conf2 = Chem.FindMolChiralCenters(mol2, includeUnassigned=True)
    
    # �ж��������Ƿ�����������
    if not conf1 or not conf2:
        return False  # ��������κ�һ������û���������ģ����ǷǶ�ӳ�칹��
        
    if len(conf1) != len(conf2):
        return False  # ��������������ͬ���������ǷǶ�ӳ�칹��
    
    # ��鲿�����������Ƿ���ͬ�������෴
    diff_count = sum(1 for (idx1, chiral1), (idx2, chiral2) in zip(conf1, conf2) if chiral1 != chiral2)
    return 0 < diff_count < len(conf1)

def find_stereoisomers(smiles_list):
    """Ѱ�Ҷ�ӳ�칹��ͷǶ�ӳ�칹��"""
    enantiomers = []
    diastereomers = []
    
    for smi1, smi2 in combinations(smiles_list, 2):  # �������
        mol1 = Chem.MolFromSmiles(smi1)
        mol2 = Chem.MolFromSmiles(smi2)
        
        if mol1 and mol2:  # ȷ�����ӺϷ�
            if is_enantiomer(mol1, mol2):
                enantiomers.append((smi1, smi2))
            elif is_diastereomer(mol1, mol2):
                diastereomers.append((smi1, smi2))
    
    return enantiomers, diastereomers
    
def save_results_to_csv(enantiomers, diastereomers, output_file):
    """�������� CSV �ļ���ȷ�� SMILES ����ȥ��"""
    data_set = set()  # ʹ�ü���ȥ��
    for smi1, smi2 in enantiomers:
        data_set.add(("Enantiomer", *sorted([smi1, smi2])))
    for smi1, smi2 in diastereomers:
        data_set.add(("Diastereomer", *sorted([smi1, smi2])))
    
    # ת��Ϊ�б����浽 CSV
    df = pd.DataFrame(list(data_set), columns=["Type", "SMILES_1", "SMILES_2"])
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
if __name__ == "__main__":
    # ���ļ��ж�ȡ SMILES �б�
    input_file = "autodl-tmp/code_final/datasets/zuhe/train_merged_smiles_with_property_scaffold_3000.txt"
    output_file = "autodl-tmp/code_final/yigouti_trainset.csv"
    
    with open(input_file, "r") as f:
        smiles_list = [line.strip() for line in f.readlines()]
    
    # �ҵ���ӳ�칹��ͷǶ�ӳ�칹��
    enantiomers, diastereomers = find_stereoisomers(smiles_list)
    
    # �������� CSV �ļ�
    save_results_to_csv(enantiomers, diastereomers, output_file)
    
    

