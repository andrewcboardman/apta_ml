raw.data <- read.csv('~/Downloads/Ilk_pool_E.txt',header = FALSE,sep=' ')

n <- length(raw.data$V1)
data <- data.frame(values=c(raw.data$V1,raw.data$V3), labels=c(rep_len(1,n),rep_len(0,n)))
data.range <- range(data$values)

output <- data.frame(thresholds=seq(from=data.range[1],to=data.range[2],length.out=100),fpr=0,tpr=0)
for (i in 1:length(output$thresholds)) {
  t <- output$thresholds[i]
  positives <- filter(data, values > t)
  output$fpr[i] <- count(filter(positives, labels == 0))/count(filter(data, labels == 0))
  output$tpr[i] <- count(filter(positives, labels == 1))/count(filter(data, labels == 1))
}

cols <- c('Pool 5'='red', 'Pool 2'='blue', 'Random'='green')
ggplot(data=raw.data) + 
  geom_histogram(aes(-V1, y=..density.., fill = 'Pool 5'), binwidth = 1, alpha = 0.5) +
  geom_histogram(aes(-V3, y=..density.., fill = 'Pool 2'), binwidth = 1, alpha = 0.5) +
  geom_histogram(aes(-V2, y=..density.., fill = 'Random'), binwidth = 1, alpha = 0.5) +
  xlab('Log probability') + ylab('Count') + ylim(c(0,0.08))
  
ggplot(data=output,aes(x=fpr,y=tpr))

