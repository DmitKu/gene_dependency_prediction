# -*- coding: utf-8 -*-
"""
Created on Sat May 30 13:50:58 2026

@author: dkuch
"""

import h5py
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns


base_path = r'C:\Users\dkuch\Documents\Blog_ideas_data\Computational\MOA_Prediction_based_on_CETSA\20251122_Model_development\GitHub_GeneDependancy_prediction'
bath_path = Path(base_path)

file_path = r'C:\Users\dkuch\Documents\Blog_ideas_data\Computational\MOA_Prediction_based_on_CETSA\20251122_Model_development\GitHub_GeneDependancy_prediction/outputs/RNA_fetures/RNA_based_features_CRISPR.csv'
file_path = Path(file_path)
# Open the file in read mode
with h5py.File(bath_path/'outputs/H5_model_data/model_H5_data.h5', 'r') as f:
    # List all groups/datasets in the root
    print("Keys in the file:", list(f.keys()))
    
    print("Contents of 'genes':", list(f['genes'].keys()))
    
    crispr_data = f['genes/crispr'][()]
    gene_ids = f['genes/gene_id'].asstr()[()]
    model_ids = f['genes/model_id'].asstr()[()]
    
    
# 2. Create the DataFrame
df_long = pd.DataFrame({
    'gene': gene_ids,
    'ModelID': model_ids,
    'crispr_H5': crispr_data
})


# Set a professional visual style
sns.set_theme(style="whitegrid")

# Create the figure
plt.figure(figsize=(10, 6))

# Plot the distribution
# 'color' and 'kde_kws' make the curve look smoother and more professional
sns.histplot(data=df_long, x='crispr_H5', kde=True, color='skyblue', edgecolor='white')

# Aesthetics for the blog
plt.title('Distribution of CRISPR Scores', fontsize=16, fontweight='bold')
plt.xlabel('CRISPR Score', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

# Remove the top and right spines for a "clean" look
sns.despine()

plt.show()


####compare with csv_file
df_raw = pd.read_csv(file_path)
df_raw.columns
df_raw = df_raw[['ModelID', 'gene', 'CRISPR']]



data_mrg = pd.merge(df_raw, df_long, on=['ModelID','gene'], how='inner')



# Using your DataFrame
df_sample = data_mrg.sample(n=10000, replace=True)
df_sample.plot.scatter(x='CRISPR', y='crispr_H5')

plt.show()



















