library(enrichvs)
data <- read.table('models/cnn/cnn_3_test_pred.txt',header=FALSE,col.names=c('labels','scores'))
data.auc <- auc(data$scores,data$labels)
data.bedroc <- bedroc(data$scores,data$labels)
data.ef <- enrichment_factor(data$scores,data$labels)

fileConn <- file('models/cnn/cnn_3_metrics.txt')
writeLines(c('AUC',data.auc,'BEDROC',data.bedroc,'Enrichment factor',data.ef),fileConn)
close(fileConn	)

