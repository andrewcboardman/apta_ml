import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from itertools import chain
# Take inputs
parser = argparse.ArgumentParser()
parser.add_argument('-i','--infile',action='store',help='Input file')
parser.add_argument('-o','--outfile',action='store',help='Output file tag')
parser.add_argument('-n','--shift_length',type=int,action='store',help='Number of bases to shift by')

# Read FASTA-formatted samples
seqs = SeqIO.parse(args.infile,'fasta')

def pad(n):
	return ''.join(np.random.choose(['A','C','G','T'],size=n))

def shift(rec,n):
	if n > 0:
		return pad(n) + rec[n:]
	elif n < 0:
		return rec[:-n] + pad(n)
	else:
		return rec
	
seqs_shift = chain(*[(shift(rec,n) for rec in seqs) for i in range(-n,n)])
SeqIO.write(seqs_shift,args.outfile,'fasta') 
