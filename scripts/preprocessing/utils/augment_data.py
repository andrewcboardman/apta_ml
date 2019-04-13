import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from itertools import chain
import argparse
import gzip
# Take inputs
parser = argparse.ArgumentParser()
parser.add_argument('-i','--infile',action='store',help='Input file')
parser.add_argument('-o','--outfile',action='store',help='Output file tag')
parser.add_argument('-n','--shift_length',type=int,action='store',help='Number of bases to shift by')
args = parser.parse_args()
def pad(n):
	return ''.join(np.random.choose(['A','C','G','T'],size=n))

def shift(rec,n):
	if n > 0:
		return pad(n) + rec[n:]
	elif n < 0:
		return rec[:-n] + pad(n)
	else:
		return rec

# Read zipped FASTA-formatted samples
with gzip.open(args.infile,'r') as f1, gzip.open(args.outfile,'w') as f2:
	seqs = SeqIO.parse(f1,'fasta')	
	seqs_shift = [(shift(rec,args.shift_length) for rec in seqs) for i in range(-args.shift_length,args.shift_length+1)]
	SeqIO.write(seqs_shift,f2,'fasta') 
