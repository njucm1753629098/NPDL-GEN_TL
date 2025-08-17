import pandas as pd
from rdkit import Chem
from collections import defaultdict
import os

# ------------------------------------------------------------------
# 1. NP-scaffold SMARTS dictionary
# ------------------------------------------------------------------
NP_SCAFFOLD_SMARTS = {
    "monoterpenoid":     "[C;R]1[C;R][C;R][C;R][C;R][C;R][C;R]1",
    "sesquiterpenoid":   "C(C)(C)C1CCCCC1",
    "diterpenoid":       "CC(C)CCC[C;R]1[C;R][C;R][C;R][C;R]1",
    "triterpenoid":      "C1C2C3CC[C;R]4C[C;R][C;R][C;R][C;R]4C3C[C;R]2C1",
    "flavonoid":         "[O;R][c;R]1[c;R][c;R]c2[c;R]c([O;R])c([O;R])c[c;R]2c1=O",
    "coumarin":          "O=C1C=CC2=CC=CC=C2O1",
    "lignan":            "C1CC2=CC=CC=C2C3=CC=CC=C31",
    "quinone":           "O=C1C(=O)C=CC=C1",
    "indole_alkaloid":   "[N;R]1[C;R]2=C[C;R]=C[C;R]=C2[C;R][C;R]1",
    "quinoline_alkaloid":"[N;R]1=C[C;R]=C2[C;R]=C[C;R]=C[C;R]=C2C=C1",
    "tropane_alkaloid":  "[N;R]1[C;R]2[C;R][C;R][C;R][C;R][C;R]2C[C;R]1",
    "steroid":           "C1C2CCC3C4CCC5=CC(=O)CCC5C4CCC3C2CC1",
}

# Pre-compile patterns
PATTERNS = {k: Chem.MolFromSmarts(v) for k, v in NP_SCAFFOLD_SMARTS.items()}

# ------------------------------------------------------------------
# 2. Core counting function with debug info
# ------------------------------------------------------------------
def count_scaffolds(df: pd.DataFrame, label: str):
    # 1. Clean column
    raw_smiles = df["SMILES"].dropna().astype(str)
    # 2. Parse mols
    mols = [Chem.MolFromSmiles(s) for s in raw_smiles]
    valid_mols = [m for m in mols if m is not None]
    invalid_num = len(raw_smiles) - len(valid_mols)
    total = max(len(valid_mols), 1)

    print(f"{label}: raw={len(raw_smiles)}, valid={len(valid_mols)}, invalid={invalid_num}")

    # 3. Count scaffolds
    counts = defaultdict(int, {k: 0 for k in NP_SCAFFOLD_SMARTS})
    for mol in valid_mols:
        for name, patt in PATTERNS.items():
            if patt and mol.HasSubstructMatch(patt):
                counts[name] += 1

    # 4. Build result dataframe
    res = [
        {"scaffold": k, "count": counts[k], "ratio": counts[k] / total}
        for k in NP_SCAFFOLD_SMARTS
    ]
    return pd.DataFrame(res)

# ------------------------------------------------------------------
# 3. File mapping
# ------------------------------------------------------------------
data = {
    "MolGPT":  "code_final/gpt_ahc_diversity_generation_10000_valid_unique.csv",
    "VAE":     "code_final/vae_ahc2_early_diversity_generation_10000_valid_unique.csv",
    "CharRNN": "code_final/char_rnn2_ahc_early_diversity_generation_10000_valid_unique.csv",
    "NP-MGDD": "code_final/gpt1_ahc_400_topk_0.25_diversity_generation_10000.csv",
}

# ------------------------------------------------------------------
# 4. Batch processing
# ------------------------------------------------------------------
os.makedirs("scaffold_counts", exist_ok=True)

for key, file_path in data.items():
    df = pd.read_csv(file_path)
    out_df = count_scaffolds(df, key)
    out_path = f"scaffold_counts/{key}_scaffold_counts.csv"
    out_df.to_csv(out_path, index=False)
    print(f"{key} -> {out_path}\n")