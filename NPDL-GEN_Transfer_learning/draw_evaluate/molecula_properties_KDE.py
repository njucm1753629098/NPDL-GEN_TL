import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.font_manager as fm

times_path = '/usr/share/fonts/truetype/msttcorefonts/times.ttf'
fm.fontManager.addfont(times_path)          
rcParams['font.family'] = 'Times New Roman'

rcParams.update({'font.size': 22})  
rcParams['axes.labelsize'] = 24     
rcParams['xtick.labelsize'] = 22   
rcParams['ytick.labelsize'] = 22    
rcParams['legend.fontsize'] = 18    


smiles_158 = pd.read_csv('NPDL-GEN_Transfer_learning/draw_evaluate/property_calculations.csv', header=None)
with open('NPDL-GEN_Transfer_learning/datasets/dataset.txt', 'r') as file:
    smiles_1386 = file.readlines()
smiles_1386 = [s.strip() for s in smiles_1386]  

# Function to calculate molecular properties
def calculate_properties(smiles_list):
    molecular_weights = []
    logps = []
    tpsas = []
    hbd = []
    hba = []
    rotatable_bonds = []
    
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            molecular_weights.append(Descriptors.MolWt(mol))
            logps.append(Descriptors.MolLogP(mol))
            tpsas.append(Descriptors.TPSA(mol))
            hbd.append(Descriptors.NumHDonors(mol))
            hba.append(Descriptors.NumHAcceptors(mol))
            rotatable_bonds.append(Descriptors.NumRotatableBonds(mol))
        else:
            molecular_weights.append(np.nan)
            logps.append(np.nan)
            tpsas.append(np.nan)
            hbd.append(np.nan)
            hba.append(np.nan)
            rotatable_bonds.append(np.nan)
    
    return pd.DataFrame({
        'MW': molecular_weights,
        'LogP': logps,
        'TPSA': tpsas,
        'HBD': hbd,
        'HBA': hba,
        'Rotatable Bonds': rotatable_bonds
    })

# Calculate properties for both datasets
properties_158 = calculate_properties(smiles_158[0].tolist())
properties_1386 = calculate_properties(smiles_1386)

# Create subplots for the properties
fig, axs = plt.subplots(2, 3, figsize=(24, 16), dpi=600)  # Increase figure size for higher resolution


sns.kdeplot(data=properties_158['MW'], label='Training Set', ax=axs[0, 0], fill=True)
sns.kdeplot(data=properties_1386['MW'], label='Generated Set', ax=axs[0, 0], fill=True)
axs[0, 0].legend()

sns.kdeplot(data=properties_158['LogP'], label='Training Set', ax=axs[0, 1], fill=True)
sns.kdeplot(data=properties_1386['LogP'], label='Generated Set', ax=axs[0, 1], fill=True)
axs[0, 1].legend()

sns.kdeplot(data=properties_158['TPSA'], label='Training Set', ax=axs[0, 2], fill=True)
sns.kdeplot(data=properties_1386['TPSA'], label='Generated Set', ax=axs[0, 2], fill=True)
axs[0, 2].legend()

sns.kdeplot(data=properties_158['HBD'], label='Training Set', ax=axs[1, 0], fill=True)
sns.kdeplot(data=properties_1386['HBD'], label='Generated Set', ax=axs[1, 0], fill=True)
axs[1, 0].legend()

sns.kdeplot(data=properties_158['HBA'], label='Training Set', ax=axs[1, 1], fill=True)
sns.kdeplot(data=properties_1386['HBA'], label='Generated Set', ax=axs[1, 1], fill=True)
axs[1, 1].legend()

sns.kdeplot(data=properties_158['Rotatable Bonds'], label='Training Set', ax=axs[1, 2], fill=True)
sns.kdeplot(data=properties_1386['Rotatable Bonds'], label='Generated Set', ax=axs[1, 2], fill=True)
axs[1, 2].legend()

# Adjust layout and save the plot
plt.tight_layout()
#plt.savefig('properties_distribution_high_res.png')
plt.savefig('properties_distribution_high_res.svg', format='svg')
# Show the plot
plt.show()
