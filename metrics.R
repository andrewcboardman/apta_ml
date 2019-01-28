library(enrichvs)
data <- read.table('models/model_xgb/model_xgb_test_pred.txt',header=FALSE,col.names=c('labels','scores'))
data.auc <- auc(data$scores,data$labels)
data.bedroc <- bedroc(data$scores,data$labels)
data.ef <- enrichment_factor(data$scores,data$labels)

fileConn <- file('xgb_metrics.txt')
writeLines(c('AUC',data.auc,'BEDROC',data.bedroc,'Enrichment factor',data.ef),fileConn)
close(fileConn	)

