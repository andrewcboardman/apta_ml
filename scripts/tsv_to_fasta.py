import pandas as pd
import csv
data = pd.read_csv('Pool2.csv',sep=',',usecols=['Aptamer Id','Sequence'])
fasta_data = '>' + data['Aptamer Id'].astype(str) + '\n' + data['Sequence']
fasta_data.to_csv('Pool2.fa',index=False,quoting=csv.QUOTE_NONE, quotechar="",  escapechar="\\")
