# -*- coding: utf-8 -*-
"""
Created on Sun May 17 14:22:51 2026

@author: dkuch
"""
def select_genes_by_variance(df, min_sd: float = 0.01) -> list[str]:
    df = df.rename(columns={"Unnamed: 0": "ModelID"}).set_index("ModelID")
    var_per_gene = df.std(axis=0, numeric_only=True)
    return var_per_gene[var_per_gene >= min_sd].index.tolist()

def select_active_inactive_genes(data_crispr,
                                  activity_threshhold,
                                  non_activity_threshhold):
    data_crispr = data_crispr.rename(columns={"Unnamed: 0": "ModelID"})
    data_crispr = data_crispr.set_index('ModelID')

    # Boolean masks
    dependent_mask = data_crispr < activity_threshhold
    nondependent_mask = data_crispr > non_activity_threshhold

    # Count per gene
    num_dependent = dependent_mask.sum(axis=0)
    num_nondependent = nondependent_mask.sum(axis=0)

    # Genes with at least 5 dependent AND 5 nondependent cell lines
    eligible_genes = (num_dependent >= 5) & (num_nondependent >= 5)

    # Extract list of eligible genes
    eligible_gene_list = eligible_genes[eligible_genes].index.tolist()
    return eligible_gene_list

import pandas as pd
from pathlib import Path

BASE_PATH   = Path("C:/Users/dkuch/Documents/Blog_ideas_data/Computational/MOA_Prediction_based_on_CETSA")
RNA_FILE    = BASE_PATH/ "public_data/DepMap/Expression/Expression_Public_25Q3_subsetted.csv"
CRISPR_FILE = BASE_PATH/ "public_data/DepMap/CRISPR/CRISPR_(DepMap_Public_25Q3+Score,_Chronos)_subsetted.csv"



RNA_data = pd.read_csv(RNA_FILE)
CRISPR_data = pd.read_csv(CRISPR_FILE)

MIN_CRISPR_SD = 0.7
print("Selecting genes by RNA variance …")
active_genes = select_genes_by_variance(RNA_FILE, min_sd=MIN_CRISPR_SD)
print(f"  Genes passing RNA variance filter: {len(active_genes):,}")

tst = [g for g in active_genes if g == 'MET']
tst

print("Selecting genes by CRISPR dyversity\n(at least 5 cell lines active and 5 cell lines inactiv) …")
active_genes_thresh = select_active_inactive_genes(data_crispr = CRISPR_data,
                                            activity_threshhold = -0.5,
                                            non_activity_threshhold = -0.3)
print(f"  Genes passing dyversity filter: {len(active_genes):,}")


common_genes = set(active_genes)&set(active_genes_thresh)

