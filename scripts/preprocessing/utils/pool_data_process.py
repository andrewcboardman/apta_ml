import pandas as pd
data = pd.read_csv('/storage/Ilk/pool234.txt',sep='\t')

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

seqs2 = data['Sequence'][data['Round2'] != 0]
ids2 = data['Round2'][data['Round2'] != 0]
recs2 = (SeqRecord(Seq(seq),id=str(count)) for (seq,count) in zip(seqs2,ids2))
SeqIO.write(recs2,'pool2_test.fa','fasta')

seqs3 = data['Sequence'][data['Round3'] != 0]
ids3 = data['Round3'][data['Round3'] != 0]
recs3 = (SeqRecord(Seq(seq),id=str(count)) for (seq,count) in zip(seqs3,ids3))
SeqIO.write(recs2,'/storage/Ilk/pool3.fa','fasta')

seqs4 = data['Sequence'][data['Round4'] != 0]
ids4 = data['Round4'][data['Round4'] != 0]
recs4 = (SeqRecord(Seq(seq),id=str(count)) for (seq,count) in zip(seqs4,ids4))
SeqIO.write(recs4,'/storage/Ilk/pool4.fa','fasta')

