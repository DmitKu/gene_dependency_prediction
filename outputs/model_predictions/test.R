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


data_gene <- data[data$gene_id=='FTSJ3',]
plot(data_gene$crispr_actual, data_gene$crispr_predicted)


data_cell <- data[data$cell_line_model_id==data$cell_line_model_id[100],]
plot(data_cell$crispr_actual, data_cell$crispr_predicted)

data_cell$rank_actual <- rank(data_cell$crispr_actual)
data_cell$rank_predicted <- rank(data_cell$crispr_predicted)

data_cell_sel <- data_cell[data_cell$crispr_predicted< -0.7,]

gene_val <- "TP53"
data_cell_sel$crispr_predicted[data_cell_sel$gene_id==
                                 data_cell_sel$gene_id[grep(gene_val,data_cell_sel$gene_id)]]
data_cell_sel$crispr_actual[data_cell_sel$gene_id==
                              data_cell_sel$gene_id[grep(gene_val,data_cell_sel$gene_id)]]


data_cell_sel$rank_predicted[data_cell_sel$gene_id==
                                 data_cell_sel$gene_id[grep(gene_val,data_cell_sel$gene_id)]]
data_cell_sel$rank_actual[data_cell_sel$gene_id==
                              data_cell_sel$gene_id[grep(gene_val,data_cell_sel$gene_id)]]



hist( data_cell$crispr_predicted)
