# coding:latin-1
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# ¶ÁÈ¡CSVÎÄ¼þ
df = pd.read_csv("NPDL-GEN_Transfer_learning/draw_evaluate/gpt1_ahc_400_topk_0.25_diversity_generation_10000_unique_with_scaffold_3000_property.csv")

# ½«sasÁÐ·Ö×é
bins = [1, 2, 3, 4, 5, 6, 7, 8]
labels = ['1-2', '2-3', '3-4', '4-5', '5-6', '6-7', '7-8']
df['sas_group'] = pd.cut(df['sas'], bins=bins, labels=labels, right=False)

# Í³¼ÆÃ¿¸ö·Ö×éµÄÊýÁ¿
sas_counts = df['sas_group'].value_counts().sort_index()

# ´´½¨»æÍ¼
plt.figure(figsize=(12/2.54, 7/2.54))


ax = sns.barplot(x=sas_counts.index, y=sas_counts.values, palette="coolwarm", edgecolor="black")

# ÃÀ»¯Ï¸½Ú
for bar in ax.patches:
    bar.set_width(0.7)
for container in ax.containers:
    ax.bar_label(container, fmt='%d', fontsize=8, padding=5)  # ÏÔÊ¾Öù×´Í¼ÉÏµÄÊý¾Ý±êÇ©


plt.xlabel("SAS", fontsize=8)
plt.ylabel("Count", fontsize=8)
plt.xticks(fontsize=8)
plt.yticks(ticks=range(0,901,100), fontsize=8)


plt.savefig("NPDL-GEN_Transfer_learning/draw_evaluate/tongji_sas_after_414.svg",dpi=300, bbox_inches='tight', pad_inches=0)

