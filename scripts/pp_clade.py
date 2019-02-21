import pandas
import numpy as np
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio.Alphabet.IUPAC import IUPACUnambiguousDNA

dtype = {'probe_name':str, 'sequence':str, 'parent_probe_name':str, 'APC_binding_score':float,
       'SE':float}

data = pandas.read_csv('data/CLADE/KnightST3R1.csv',header=4,dtype=dtype,na_values='.')
data_tidy = data.dropna()
seqs_train = (Seq(sequence,alphabet=IUPACUnambiguousDNA) for sequence in data_tidy['sequence'])
records_train = (SeqRecord(seq,id=id_train) for (seq,id_train) in zip(seqs_train,data_tidy['probe_name']))
SeqIO.write(records_train,'data/CLADE/seqs_train.fasta','fasta')
np.savetxt('data/CLADE/scores_train.txt',data_tidy['APC_binding_score'].values.astype(float))

dtype.pop('parent_probe_name')
data2 = pandas.read_csv('data/CLADE/KnightST5R1.csv',header=3,dtype=dtype)
data2_tidy = data2.dropna()
seqs_test = (Seq(sequence,alphabet=IUPACUnambiguousDNA) for sequence in data2_tidy['sequence'])
records_test = (SeqRecord(seq,id=id_test) for (seq,id_test) in zip(seqs_test,data2_tidy['probe_name']))
SeqIO.write(records_test,'data/CLADE/seqs_test.fasta','fasta')
np.savetxt('data/CLADE/scores_test.txt',data2_tidy['APC_binding_score'].values.astype(float))