# src/utils_correlation.R
# Core functions for high-performance DepMap correlation analysis

library(data.table)
library(dplyr)
library(parallel)

#' Fast Spearman Correlation for High-Dimensional Genomics
cor_crispr_RNA_fast <- function(data_RNA_fn, data_crispr_fn, gene_val) {
  
  # Subset CRISPR gene and align with RNA data
  data_crispr_sub <- data_crispr_fn[, c('ModelID', gene_val), with = FALSE]
  setnames(data_crispr_sub, gene_val, "CRISPR")
  
  # Fast inner join
  data_comb <- merge(data_crispr_sub, data_RNA_fn, by = "ModelID", all = FALSE)
  data_comb <- data_comb[!is.na(CRISPR)]
  
  # Convert RNA to matrix for computational efficiency
  crispr_vec <- data_comb$CRISPR
  rna_mat <- as.matrix(data_comb[, !c("ModelID", "CRISPR"), with = FALSE])
  
  # Parallel Cluster Setup
  n_cores <- max(1, detectCores() - 1)
  cl <- makeCluster(n_cores)
  on.exit(stopCluster(cl)) # Ensure RAM is freed on exit
  
  clusterExport(cl, c("crispr_vec", "rna_mat"), envir = environment())
  
  # Execute Spearman tests across all RNA features
  results <- parLapply(cl, 1:ncol(rna_mat), function(i) {
    rna_col <- rna_mat[, i]
    valid_idx <- !is.na(crispr_vec) & !is.na(rna_col)
    
    if (sum(valid_idx) < 10) return(list(cor = NA, pval = NA))
    
    test <- cor.test(crispr_vec[valid_idx], rna_col[valid_idx], 
                     method = "spearman", exact = FALSE)
    list(cor = test$estimate, pval = test$p.value)
  })
  
  # Return structured results
  return(data.table(
    spearman = sapply(results, function(x) x$cor),
    p.value = sapply(results, function(x) x$pval),
    crispr_gene = gene_val,
    rna_gene = colnames(rna_mat)
  ))
}

#' Batch Processor with Checkpointing
run_cor_multiple_genes_batch <- function(gene_list, data_RNA_fn, data_crispr_fn, output_dir, batch_size = 10) {
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
  
  n_batches <- ceiling(length(gene_list) / batch_size)
  
  for (i in 1:n_batches) {
    cat(sprintf("\nProcessing Batch %d of %d...\n", i, n_batches))
    
    batch_genes <- gene_list[((i-1)*batch_size + 1):min(i*batch_size, length(gene_list))]
    
    batch_results <- lapply(batch_genes, function(gene) {
      tryCatch(cor_crispr_RNA_fast(data_RNA_fn, data_crispr_fn, gene), 
               error = function(e) {
                 warning(paste("Error in gene", gene, ":", e$message))
                 return(NULL)
               })
    })
    
    # Save checkpoint as RDS
    final_batch <- rbindlist(batch_results[!sapply(batch_results, is.null)])
    if (nrow(final_batch) > 0) {
      saveRDS(final_batch, file.path(output_dir, paste0("batch_", i, ".rds")))
    }
  }
}