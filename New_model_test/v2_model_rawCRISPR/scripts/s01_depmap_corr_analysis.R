# run_depmap_analysis.R
# Main execution script for Gene Dependency Prediction Project

# 1. Paths Configuration (Tuned for your local setup)
BASE_PATH   <- "C:/Users/dkuch/Documents/Blog_ideas_data/Computational/MOA_Prediction_based_on_CETSA"
src_file    <- file.path(BASE_PATH, "20251122_Model_development/GitHub_GeneDependancy_prediction/src/utils_correlation.R")
RNA_FILE    <- file.path(BASE_PATH, "public_data/DepMap/Expression/Expression_Public_25Q3_subsetted.csv")
CRISPR_FILE <- file.path(BASE_PATH, "public_data/DepMap/CRISPR/CRISPR_(DepMap_Public_25Q3+Score,_Chronos)_subsetted.csv")
OUT_DIR     <- file.path(BASE_PATH, "Analysis_data/tst")

# 2. Load Functions
source(src_file)

# 3. Load Data with High-Speed fread
message("Loading DepMap datasets...")
data_RNA    <- fread(RNA_FILE)
data_crispr <- fread(CRISPR_FILE)

# Standardize IDs
setnames(data_RNA, 1, "ModelID")
setnames(data_crispr, 1, "ModelID")
names(data_crispr) <- gsub('\\.\\..+', '', names(data_crispr))

# 4. Filter Genes (Quality Control)
all_genes <- intersect(names(data_RNA), names(data_crispr))
all_genes <- all_genes[all_genes != "ModelID"]

# Only use genes with complete CRISPR profiles (0 NA)
complete_crispr <- names(data_crispr)[colSums(is.na(data_crispr)) == 0]
gene_list <- intersect(all_genes, complete_crispr)

# 5. Run Batch Correlation
message(sprintf("Starting analysis for %d genes...", length(gene_list)))
run_cor_multiple_genes_batch(
  gene_list = gene_list,
  data_RNA_fn = data_RNA,
  data_crispr_fn = data_crispr,
  output_dir = OUT_DIR,
  batch_size = 10
)

message("Analysis Complete. Results stored in: ", OUT_DIR)