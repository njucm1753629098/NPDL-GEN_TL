# coding:latin-1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import  SimpleImputer
from matplotlib import rcParams, font_manager

if __name__ == '__main__':
    

    
    en_font = {'family':'Times New Roman','size':8}
    
    
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 8
    plt.rcParams['axes.unicode_minus'] = False  
    
    
    generation_df = pd.read_csv('code_final/pretrain_gpt1_generation_10000_unique_with_property.csv')
    train_df = pd.read_csv('code_final/datasets/zuhe/train_merged_smiles_with_property.csv')
    
    
    generation_df = generation_df.sample(n=8000, random_state=42)
    train_df = train_df.sample(n=8000, random_state=42)


    combined_df = pd.concat([generation_df, train_df])
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(combined_df[['logp', 'mw', 'tpsa', 'hbd', 'hba', 'rotbonds', 'qed', 'sas']].values)


    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)


    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)


    plt.figure(figsize=(12/2.54, 7/2.54))
    plt.scatter(X_pca[:len(generation_df), 0], X_pca[:len(generation_df), 1], c='#FCB373', label='gpt1 generation', s=6) #14
    plt.scatter(X_pca[len(train_df):, 0], X_pca[len(train_df):, 1], alpha=0.5, c='#7AADD2', label='training set', s=6)
    
    
    plt.title('')
    plt.xlabel('Principal Component 1',fontproperties=en_font)
    plt.ylabel('Principal Component 2',fontproperties=en_font)
    plt.legend(prop=en_font, fontsize=8)
    plt.grid(False)
    plt.savefig('code_final/PCA_gpt1.png', bbox_inches='tight', dpi=300)



 