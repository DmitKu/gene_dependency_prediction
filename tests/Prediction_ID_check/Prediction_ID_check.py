# -*- coding: utf-8 -*-
"""
Created on Sat May 30 18:53:53 2026

@author: dkuch
"""
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import joblib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns


def plot_scatter_lmfit(data ,
                       x = 'CRISPR',
                       y = 'crispr_predicted'):
    # 1. Use hexbin for massive point density - it's much faster than scatter()
    plt.figure(figsize=(8, 6))
    plt.scatter(
        data[ x ], 
        data[y], 
        alpha=0.05, 
        s=1, 
        color='darkgrey', 
        rasterized=True 
    )

    # 2. Calculate linear regression using numpy (much faster than regplot)
    # This uses polyfit to get the slope (m) and intercept (c)
    m, c = np.polyfit(data[ x ], data[y], 1)

    # Generate line points
    x_vals = np.array([data[ x ].min(), data[ x ].max()])
    y_vals = m * x_vals + c

    # Plot the line
    plt.plot(x_vals, y_vals, color='red', linestyle='--', linewidth=2, label=f'Trend: y={m:.2f}x + {c:.2f}')

    # 3. Calculate Pearson r quickly
    r, _ = stats.pearsonr(data[ x ], data['crispr_predicted'])
    plt.text(0.05, 0.95, f'Pearson r = {r:.2f}', transform=plt.gca().transAxes, fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.5))

    plt.title(f"{x} vs. {y} (700k points)")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.legend()
    plt.show()



#############################
#############################

DEPMAP_BASE = Path(
    r"C:\Users\dkuch\Documents\Blog_ideas_data\Computational"
    r"\MOA_Prediction_based_on_CETSA\public_data\DepMap"
)



CRISPR_FILE      = DEPMAP_BASE / "CRISPR"     / "CRISPR_(DepMap_Public_25Q3+Score,_Chronos)_subsetted.csv"

PREDICTION_FILE  = Path(r'C:\Users\dkuch\Documents\Blog_ideas_data\Computational\MOA_Prediction_based_on_CETSA\20251122_Model_development\GitHub_GeneDependancy_prediction\outputs\model_predictions\predictions_test.csv')

TRANSFORMER_FILE = Path(r'C:\Users\dkuch\Documents\Blog_ideas_data\Computational\MOA_Prediction_based_on_CETSA\20251122_Model_development\GitHub_GeneDependancy_prediction\outputs\RNA_fetures\chronos_quantile_transformer.pkl')

crispr_wide = pd.read_csv(DEPMAP_BASE/CRISPR_FILE)
prediction_df = pd.read_csv(PREDICTION_FILE)


crispr_wide = crispr_wide.rename(columns={'Unnamed: 0': 'cell_line_model_id'})

df_long = crispr_wide.reset_index().melt(
    id_vars='cell_line_model_id', 
    var_name='gene_id', 
    value_name='CRISPR'
)



qt = joblib.load(TRANSFORMER_FILE)



prediction_df_mrg = prediction_df.merge( df_long,
                          on = ['cell_line_model_id','gene_id'],
                          how='left')

prediction_df_mrg['crispr_actual_transformed'] = qt.inverse_transform(prediction_df_mrg['crispr_actual'].values.reshape(-1, 1)).squeeze()

##### plot

# Assuming x and y are your data lists or arrays
prediction_df_mrg.columns
'''
Index(['sample_idx', 'cell_line_model_id', 'gene_id', 'crispr_actual',
       'crispr_predicted', 'residual', 'CRISPR', 'crispr_actual_transformed'],
      dtype='object')
'''

plot_scatter_lmfit(prediction_df_mrg ,
                       x = 'CRISPR',
                       y = 'crispr_actual_transformed')

plot_scatter_lmfit(prediction_df_mrg ,
                       x = 'CRISPR',
                       y = 'crispr_predicted')

# Histogram with a Kernel Density Estimate (KDE) curve
sns.histplot(x=prediction_df_mrg['CRISPR'], kde=True)
plt.show()

sns.histplot(x=prediction_df_mrg['crispr_actual'], kde=True)
plt.show()

sns.histplot(x=prediction_df_mrg['crispr_predicted'], kde=True)
plt.show()

sns.histplot(x=prediction_df_mrg['crispr_actual_transformed'], kde=True)
plt.show()