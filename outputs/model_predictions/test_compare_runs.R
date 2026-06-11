library(rstudioapi)
library(dplyr)
library(ggplot2)


setwd('C:/Users/dkuch/Documents/Blog_ideas_data/Computational/MOA_Prediction_based_on_CETSA/20251122_Model_development/GitHub_GeneDependancy_prediction/data/2024_Shi_etal')
she_data <- read.csv('figure_1h.csv')
names(she_data)[1] = 'gene_id'

script_dir <- dirname(getActiveDocumentContext()$path)

setwd(script_dir)
names(data2)
# Now paths are relative to the script location
data2 <- read.csv("try2_predictions_test.csv")
data2$run <- 'run 2'
data1 <- read.csv("try1_predictions_test.csv")
data1$run <- 'run 1'
data0 <- read.csv("try0_predictions_test.csv")
data0$run <- 'run 0'

data_comb <- rbind(data0,data1,data2)
names(data_comb)

for (gene_val in c('EGFR', 'ERBB2', 'KRAS', 'BRAF', 'CDK4', 
                   'CDK6', 'MTOR', 'FGFR1', 'FGFR2', 'FGFR3',
                   'TUBB')){
  #gene_val = 'EGFR'
  print(gene_val)
  data_gene <- data_comb[data_comb$gene_id==gene_val,]
  
  # 1. Prepare data and calculate stats per facet
  stats_data <- data_gene %>%
    group_by(run) %>%
    summarise(
      r_val = cor(crispr_actual, crispr_predicted, use = "complete.obs"),
      p_val = cor.test(crispr_actual, crispr_predicted)$p.value,
      # Define label positions (using max/min of current facet)
      label_x = min(crispr_actual, na.rm = TRUE),
      label_y = max(crispr_predicted, na.rm = TRUE),
      label_text = paste0("r = ", round(r_val, 3), 
                          "\np = ", format.pval(p_val, digits = 2))
    )
  
  # 2. Plot with geom_text
  p = ggplot(data_gene, aes(x = crispr_actual, y = crispr_predicted)) +
    geom_point(alpha = 0.6, color = "steelblue") +
    geom_smooth(method = "lm", color = "darkred", se = TRUE) +
    # Add the labels from the stats_data frame
    geom_text(data = stats_data, 
              aes(x = label_x, y = label_y, label = label_text), 
              hjust = 0, vjust = 1, size = 3, inherit.aes = FALSE) +
    labs(title = paste0(gene_val, " gene"),
         x = "Actual CRISPR",
         y = "Predicted CRISPR") +
    theme_bw() + 
    facet_grid(~run)
  
  print(p)
  
}


library(dplyr)
names(data_comb)
data_corr <- data_comb %>%
  group_by(gene_id,run) %>%
  summarise(
    pearson = cor(crispr_actual, crispr_predicted, use = "complete.obs", method = "pearson"),
    n = sum(!is.na(crispr_actual) & !is.na(crispr_predicted))
  )



data_corr_she <- left_join(data_corr, she_data, by ='gene_id')
data_corr_she <- data_corr_she[!is.na(data_corr_she$Expression.only.CV.cor),]
names(data_corr_she)

# 1. Prepare data and calculate stats per facet
stats_data <- data_corr_she %>%
  group_by(run) %>%
  summarise(
    r_val = cor(Expression.only.CV.cor, pearson, use = "complete.obs"),
    p_val = cor.test(Expression.only.CV.cor, pearson)$p.value,
    # Define label positions (using max/min of current facet)
    label_x = min(Expression.only.CV.cor, na.rm = TRUE),
    label_y = max(pearson, na.rm = TRUE),
    label_text = paste0("r = ", round(r_val, 3), 
                        "\np = ", format.pval(p_val, digits = 2))
  )

 ggplot(data_corr_she, aes(x = Expression.only.CV.cor, y = pearson)) +
  geom_point(alpha = 0.6, color = "steelblue") +
  geom_smooth(method = "lm", color = "darkred", se = TRUE) +
  geom_text(data = stats_data,
            aes(x = label_x, y = label_y, label = label_text),
            hjust = 0, vjust = 1, size = 3, inherit.aes = FALSE) +
  labs(title = paste0(gene_val, " gene"),
       x = "Actual CRISPR",
       y = "Predicted CRISPR") +
  theme_bw() + 
  facet_grid(~run)
