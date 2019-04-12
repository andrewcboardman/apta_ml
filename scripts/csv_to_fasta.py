import pandas as pd 
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

data = pd.read_csv('Hoinka_samples.csv',sep=',')
seqs = data['Consensus sequence']
SeqIO.write((SeqRecord(Seq(x)) for x in seqs),'hoinka_samples.fasta','fasta')