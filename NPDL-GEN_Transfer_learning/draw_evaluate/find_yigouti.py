# coding:latin-1
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from itertools import combinations
import pandas as pd

def is_enantiomer(mol1, mol2):
    """ÅÐ¶ÏÁ½·Ö×ÓÊÇ·ñÎª¶ÔÓ³Òì¹¹Ìå"""
    
    # ÅÐ¶ÏÁ½·Ö×ÓµÄ·Ö×ÓÊ½ÊÇ·ñÏàÍ¬
    formula1 = rdMolDescriptors.CalcMolFormula(mol1)
    formula2 = rdMolDescriptors.CalcMolFormula(mol2)
    
    if formula1 != formula2:
        return False # Èç¹û·Ö×ÓÊ½²»Í¬£¬¾Í²»ÊÇ¶ÔÓ³Òì¹¹Ìå
    
    # ²éÕÒÊÖÐÔÖÐÐÄ
    conf1 = Chem.FindMolChiralCenters(mol1, includeUnassigned=True) 
    # ¼ÆËãÊÖÐÔÖÐÐÄÊý£¬²¢·µ»ØÎ»ÖÃºÍ(Ë÷Òý)ºÍÊÖÐÔ±êÖ¾£¬·Ö×Ó 1 µÄÊÖÐÔÖÐÐÄ£º[(1, 'S')]£¬·Ö×Ó 2 µÄÊÖÐÔÖÐÐÄ£º[(1, 'R')]£¬¶ÔÓÚ¶ÔÓ³Òì¹¹Ìå£¬ÊÖÐÔ±êÖ¾ 'S' ºÍ 'R' ÊÇÏà·´µÄ£¬Òò´Ë·ûºÏÌõ¼þ¡£¶ÔÓ³Òì¹¹Ìå±ØÐëÔÚÏàÍ¬µÄÎ»ÖÃÉÏ¾ßÓÐÊÖÐÔÖÐÐÄ
    conf2 = Chem.FindMolChiralCenters(mol2, includeUnassigned=True)
    
    # ÅÐ¶ÏÁ½·Ö×ÓÊÇ·ñº¬ÓÐÊÖÐÔÖÐÐÄ
    if not conf1 or not conf2:
        return False  # Èç¹ûÆäÖÐÈÎºÎÒ»¸ö·Ö×ÓÃ»ÓÐÊÖÐÔÖÐÐÄ£¬Ôò²»ÊÇ¶ÔÓ³Òì¹¹Ìå
    
    # ÅÐ¶ÏÁ½·Ö×ÓÊÖÐÔÖÐÐÄÊýÁ¿ÊÇ·ñÏàÍ¬
    if len(conf1) != len(conf2):
        return False  # ÊÖÐÔÖÐÐÄÊýÁ¿²»Í¬£¬¿Ï¶¨²»ÊÇ¶ÔÓ³Òì¹¹Ìå
    
    # ¼ì²éÊÖÐÔÖÐÐÄÊÇ·ñÍêÈ«Ïà·´
    for (idx1, chiral1), (idx2, chiral2) in zip(conf1, conf2):
        if idx1 != idx2 or chiral1 == chiral2:  # ÊÖÐÔÖÐÐÄÏàÍ¬»òÎ»ÖÃ²»Æ¥Åä
            return False
    return True

def is_diastereomer(mol1, mol2):
    """ÅÐ¶ÏÁ½·Ö×ÓÊÇ·ñÎª·Ç¶ÔÓ³Òì¹¹Ìå"""
    
    # ÅÐ¶ÏÁ½·Ö×ÓµÄ·Ö×ÓÊ½ÊÇ·ñÏàÍ¬
    formula1 = rdMolDescriptors.CalcMolFormula(mol1)
    formula2 = rdMolDescriptors.CalcMolFormula(mol2)
    
    if formula1 != formula2:
        return False # Èç¹û·Ö×ÓÊ½²»Í¬£¬¾Í²»ÊÇ·Ç¶ÔÓ³Òì¹¹Ìå
        
    conf1 = Chem.FindMolChiralCenters(mol1, includeUnassigned=True)
    conf2 = Chem.FindMolChiralCenters(mol2, includeUnassigned=True)
    
    # ÅÐ¶ÏÁ½·Ö×ÓÊÇ·ñº¬ÓÐÊÖÐÔÖÐÐÄ
    if not conf1 or not conf2:
        return False  # Èç¹ûÆäÖÐÈÎºÎÒ»¸ö·Ö×ÓÃ»ÓÐÊÖÐÔÖÐÐÄ£¬Ôò²»ÊÇ·Ç¶ÔÓ³Òì¹¹Ìå
        
    if len(conf1) != len(conf2):
        return False  # ÊÖÐÔÖÐÐÄÊýÁ¿²»Í¬£¬²»¿ÉÄÜÊÇ·Ç¶ÔÓ³Òì¹¹Ìå
    
    # ¼ì²é²¿·ÖÊÖÐÔÖÐÐÄÊÇ·ñÏàÍ¬£¬²¿·ÖÏà·´
    diff_count = sum(1 for (idx1, chiral1), (idx2, chiral2) in zip(conf1, conf2) if chiral1 != chiral2)
    return 0 < diff_count < len(conf1)

def find_stereoisomers(smiles_list):
    """Ñ°ÕÒ¶ÔÓ³Òì¹¹ÌåºÍ·Ç¶ÔÓ³Òì¹¹Ìå"""
    enantiomers = []
    diastereomers = []
    
    for smi1, smi2 in combinations(smiles_list, 2):  # Á½Á½×éºÏ
        mol1 = Chem.MolFromSmiles(smi1)
        mol2 = Chem.MolFromSmiles(smi2)
        
        if mol1 and mol2:  # È·±£·Ö×ÓºÏ·¨
            if is_enantiomer(mol1, mol2):
                enantiomers.append((smi1, smi2))
            elif is_diastereomer(mol1, mol2):
                diastereomers.append((smi1, smi2))
    
    return enantiomers, diastereomers
    
def save_results_to_csv(enantiomers, diastereomers, output_file):
    """±£´æ½á¹ûµ½ CSV ÎÄ¼þ£¬È·±£ SMILES ÅÅÐò²¢È¥ÖØ"""
    data_set = set()  # Ê¹ÓÃ¼¯ºÏÈ¥ÖØ
    for smi1, smi2 in enantiomers:
        data_set.add(("Enantiomer", *sorted([smi1, smi2])))
    for smi1, smi2 in diastereomers:
        data_set.add(("Diastereomer", *sorted([smi1, smi2])))
    
    # ×ª»»ÎªÁÐ±í²¢±£´æµ½ CSV
    df = pd.DataFrame(list(data_set), columns=["Type", "SMILES_1", "SMILES_2"])
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
if __name__ == "__main__":
    # ´ÓÎÄ¼þÖÐ¶ÁÈ¡ SMILES ÁÐ±í
    input_file = "autodl-tmp/code_final/datasets/zuhe/train_merged_smiles_with_property_scaffold_3000.txt"
    output_file = "autodl-tmp/code_final/yigouti_trainset.csv"
    
    with open(input_file, "r") as f:
        smiles_list = [line.strip() for line in f.readlines()]
    
    # ÕÒµ½¶ÔÓ³Òì¹¹ÌåºÍ·Ç¶ÔÓ³Òì¹¹Ìå
    enantiomers, diastereomers = find_stereoisomers(smiles_list)
    
    # ±£´æ½á¹ûµ½ CSV ÎÄ¼þ
    save_results_to_csv(enantiomers, diastereomers, output_file)
    
    

