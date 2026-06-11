library(rstudioapi)
library(dplyr)
library(ggplot2)

script_dir <- dirname(getActiveDocumentContext()$path)
setwd(script_dir)

# Now paths are relative to the script location
data <- read.csv("try1_predictions_test.csv")
data$abs_residual <- abs(data$residual)

mean_res <- data %>%
            group_by(gene_id) %>%
            summarize(mean_value = mean(abs_residual, na.rm = TRUE),
                      sd_value = sd(abs_residual, na.rm = TRUE))
#genes EGFR, KRAS, BRAF, TP53, MYC. PIK3AC
gene_val = 'KRAS'

for (gene_val in c('EGFR', 'ERBB2', 'KRAS', 'BRAF', 'CDK4', 
                   'CDK6', 'MTOR', 'FGFR1', 'FGFR2', 'FGFR3',
                   'TUBB')){
  #gene_val = 'NTRK'
  print(gene_val)
  data_gene <- data[data$gene_id==gene_val,]
  data_gene$rank_actual <- rank(data_gene$crispr_actual)
  data_gene$rank_predicted <- rank(data_gene$crispr_predicted)
  
  cor_val <- cor.test(data_gene$crispr_actual, data_gene$crispr_predicted)
  r_label <- paste0("Pearson r = ", round(cor_val$estimate, 3), 
                    "\n(p < ", format.pval(cor_val$p.value, digits = 2), ")")
  
  # 2. Create the ggplot
  p = ggplot(data_gene, aes(x = crispr_actual, y = crispr_predicted)) +
    geom_point(alpha = 0.6, color = "steelblue") +
    geom_smooth(method = "lm", color = "darkred", se = TRUE) + # Optional: Adds trend line
    labs(title = paste0(gene_val, " gene [model 1]"),
         x = "Actual CRISPR",
         y = "Predicted CRISPR") +
    annotate("text", x = min(data_gene$crispr_actual), y = max(data_gene$crispr_predicted), 
             label = r_label, hjust = 0, vjust = 1, size = 4) +
    theme_bw()
  print(p)
  
}

plot(data_gene$rank_actual, data_gene$rank_predicted,
     main = paste0(gene_val, " gene [rank]"))

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
