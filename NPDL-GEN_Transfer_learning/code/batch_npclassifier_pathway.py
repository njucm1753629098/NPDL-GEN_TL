#!/usr/bin/env python3

import os
import json
import time
import requests
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


API_URL     = "https://npclassifier.gnps2.org/classify"
INPUT_FILES = ["dataset.csv"] ##You can change file here
CHUNK_SIZE  = 500
MAX_WORKERS = 16
SLEEP_SEC   = 0.12
BASE_OUT    = "results"
os.makedirs(BASE_OUT, exist_ok=True)

PATHWAY_CLASSES = [
    "Alkaloids", "Terpenoids", "Polyketides",
    "Shikimates and Phenylpropanoids", "Amino Acids and Peptides",
    "Carbohydrates", "Fatty acids", "Nucleosides",
    "Steroids", "Carotenoids", "Mixed", "Other", "Glycosides"
]
ALL_CATEGORIES = PATHWAY_CLASSES + ["Unclassified"]


def fetch_pathway(smiles: str) -> str:
    try:
        resp = requests.get(API_URL, params={"smiles": smiles}, timeout=15)
        if resp.status_code == 200:
            pathway = resp.json().get("pathway_results", [None])[0]
            return pathway if pathway else "Unclassified"
    except Exception:
        pass
    return "Unclassified"

def worker(smi):
    pathway = fetch_pathway(smi)
    time.sleep(SLEEP_SEC)
    return smi, pathway

def process_one_file(fname: str):
    base_name = os.path.splitext(os.path.basename(fname))[0]
    out_dir   = os.path.join(BASE_OUT, base_name)
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(fname):
        return

    df = pd.read_csv(fname)
    if "SMILES" not in df.columns:
        return

    cache_file = os.path.join(out_dir, f"{base_name}_pathway_cache.json")
    pathway_map = {}
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            raw = json.load(f)
        pathway_map = {k: v for k, v in raw.items() if isinstance(v, str)}

    smiles_list = df["SMILES"].dropna().astype(str).tolist()
    todo_idx = [(i, s) for i, s in enumerate(smiles_list) if s not in pathway_map]



    def save_chunk(buffer):
        start = buffer[0][0] + 1
        end   = buffer[-1][0] + 1
        df_buf = pd.DataFrame(buffer, columns=["index", "SMILES", "Pathway"])
        chunk_path = os.path.join(out_dir, f"{base_name}_{start}_{end}.csv")
        df_buf.to_csv(chunk_path, index=False, encoding="utf-8")

    buffer = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
        future_to_idx = {exe.submit(worker, s): (i, s) for i, s in todo_idx}
        pbar = tqdm(total=len(todo_idx), desc=f"Processing {base_name}")
        for fut in as_completed(future_to_idx):
            idx, smi = future_to_idx[fut]
            _, pathway = fut.result()
            pathway_map[smi] = pathway
            buffer.append((idx, smi, pathway))
            pbar.update(1)

            if len(buffer) >= CHUNK_SIZE:
                save_chunk(buffer)
                buffer.clear()

            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(pathway_map, f, ensure_ascii=False, indent=2)

        if buffer:
            save_chunk(buffer)
        pbar.close()

    df["Pathway"] = df["SMILES"].map(pathway_map).fillna("Unclassified")
    full_path = os.path.join(out_dir, f"{base_name}_with_pathway.csv")
    df[["SMILES", "Pathway"]].to_csv(full_path, index=False, encoding="utf-8")

    counts = df["Pathway"].value_counts()
    counts = counts.reindex(ALL_CATEGORIES, fill_value=0)
    stats = pd.DataFrame({
        "Pathway": counts.index,
        "Count": counts.values,
        "Proportion": (counts.values / counts.sum()).round(4)
    })
    stats_path = os.path.join(out_dir, f"{base_name}_pathway_stats.csv")
    stats.to_csv(stats_path, index=False, encoding="utf-8")
    print(stats, "\n")

def main():
    for file in INPUT_FILES:
        process_one_file(file)

if __name__ == "__main__":
    main()