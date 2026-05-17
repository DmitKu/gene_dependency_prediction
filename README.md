# **Latent Gene Dependency Prediction via Manifold Clustering
This project implements a hybrid R/Python pipeline to predict DepMap Chronos scores using biologically informed latent features. Instead of relying on raw expression matrices, the model uses manifold‑derived gene clusters to improve interpretability and predictive power.

🔬 Scientific Logic
Standard gene‑by‑gene modeling often misses functional redundancy across pathways. This pipeline addresses that by combining correlation structure, manifold learning, and attention‑based modeling.

Correlation Mapping — Identify transcriptional correlates of gene dependency

Manifold Learning — Reduce the dependency–expression space

Latent Feature Engineering — Group genes into ~1,850 functional clusters

Attention‑Based Prediction — Learn which gene modules drive cell‑line‑specific vulnerabilities

##Project Structure

depmap-attention/
├── data/                               # Raw and processed DepMap 25Q3 data
├── src/
│   ├── utils_cor.R                     # Optimized Spearman correlation functions
│   ├── clustering.py                   # UMAP + DBscan clustering logic
│   └── model.py                        # PyTorch Attention Network architecture
├── scripts/
│   ├── 01_run_depmap_corr_analysis.R   # R driver for correlation computation
│   └── 02_train_nn.py                  # Feature engineering + model training
├── environment.yml                     # Conda environment configuration
└── README.md


##Getting Started
1. Prerequisites
Place the following DepMap 25Q3 datasets into data/raw/:

Expression_Public_25Q3_subsetted.csv [too big, not provided]

CRISPR_Chronos_subsetted.csv [too big, not provided]

2. Environment Setup
bash
conda env create -f environment.yml
conda activate depmap-env

3. Running the Pipeline
Step 1 — Correlation Calculation (R)
Computes Spearman correlations between CRISPR sensitivity and RNA expression.

bash
Rscript scripts/01_run_cor.R
Step 2 — Train the Model (Python)
Runs UMAP, Descant clustering, and trains the Attention‑based Neural Network.

bash
python scripts/02_train_nn.py
📊 Key Methodology: Attention Mechanism
The Attention Network extracts cluster‑level feature importance, enabling biological interpretation.
By examining attention weights, you can identify which functional gene modules drive predictions for specific cancer lineages.

📈 Results & Medium Blog
A full breakdown of biological insights and model performance is available here:
Medium Article