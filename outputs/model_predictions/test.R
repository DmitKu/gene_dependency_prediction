library(rstudioapi)
library(dplyr)

script_dir <- dirname(getActiveDocumentContext()$path)
setwd(script_dir)

# Now paths are relative to the script location
data <- read.csv("predictions_test.csv")
data$abs_residual <- abs(data$residual)

mean_res <- data %>%
            group_by(gene_id) %>%
            summarize(mean_value = mean(abs_residual, na.rm = TRUE),
                      sd_value = sd(abs_residual, na.rm = TRUE))
#genes EGFR, KRAS, BRAF, TP53, MYC. PIK3AC
gene_val = 'BRAF'
data_gene <- data[data$gene_id==gene_val,]
data_gene$rank_actual <- rank(data_gene$crispr_actual)
data_gene$rank_predicted <- rank(data_gene$crispr_predicted)

plot(data_gene$crispr_actual, data_gene$crispr_predicted,
     main = paste0(gene_val, " gene"))
plot(data_gene$rank_actual, data_gene$rank_predicted,
     main = paste0(gene_val, " gene [rank]"))


cor.test(data_gene$crispr_actual, data_gene$crispr_predicted)
cor.test(data_gene$crispr_actual, data_gene$crispr_predicted, method = 'spearman')
cor.test(data_gene$crispr_actual, data_gene$crispr_predicted, method = 'kendall')

cor.test(data_gene$rank_actual, data_gene$rank_predicted)


library(dplyr)
names(data)
data_corr <- data %>%
  group_by(gene_id) %>%
  summarise(
    pearson = cor(crispr_actual, crispr_predicted, use = "complete.obs", method = "pearson"),
    n = sum(!is.na(crispr_actual) & !is.na(crispr_predicted))
  )

mean(data_corr$pearson)

########################
# cell model centric

data$cell_line_model_id[grep('264',data$cell_line_model_id)]

data_cell <- data[data$cell_line_model_id==data$cell_line_model_id[100],]
data_cell <- data[data$cell_line_model_id=='ACH-000114',]

plot(data_cell$crispr_actual, data_cell$crispr_predicted)

data_cell$rank_actual <- rank(data_cell$crispr_actual)
data_cell$rank_predicted <- rank(data_cell$crispr_predicted)

data_cell_sel <- data_cell[data_cell$crispr_predicted< -0.7,]

gene_val <- "KRAS"
data_cell_sel$crispr_predicted[data_cell_sel$gene_id==
                                 data_cell_sel$gene_id[grep(gene_val,data_cell_sel$gene_id)]]
data_cell_sel$crispr_actual[data_cell_sel$gene_id==
                              data_cell_sel$gene_id[grep(gene_val,data_cell_sel$gene_id)]]


data_cell_sel$rank_predicted[data_cell_sel$gene_id==
                                 data_cell_sel$gene_id[grep(gene_val,data_cell_sel$gene_id)]]
data_cell_sel$rank_actual[data_cell_sel$gene_id==
                              data_cell_sel$gene_id[grep(gene_val,data_cell_sel$gene_id)]]



hist( data_cell$crispr_predicted)
