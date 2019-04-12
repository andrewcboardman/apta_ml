library(enrichvs)

metrics <- function(infile, outfile) {
  data <- read.table(infile,sep=',',header=TRUE)
  data.auc <- 0#auc(data$predictions,data$labels)
  data.bedroc <- bedroc(data$predictions,data$labels)
  data.ef <- enrichment_factor(data$predictions,data$labels)
  
  fileConn <- file(outfile)
  writeLines(c('AUC',data.auc,'BEDROC',data.bedroc,'Enrichment factor',data.ef),fileConn)
  close(fileConn)
}
metrics('IgE_deep_eval.csv','IgE_deep_metrics.txt')
metrics('Ilk_Pool_deep_eval.csv','Ilk_Pool_deep_metrics.txt')
metrics('Ilk_Pool_shallow_eval.csv','Ilk_Pool_shallow_metrics.txt')
metrics('Ilk_fit_deep_eval.csv','Ilk_fit_deep_metrics.txt')
