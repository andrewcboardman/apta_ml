data = read.table('data/minus_encode.txt')
data_stds <- rep(0,120)
for (i in 1:120) {
  data_stds[i] <- std(data[,i])
}

