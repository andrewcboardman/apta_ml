from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import numpy as np

def MonteCarloStep(x,L,J,h):
	pos = np.random.randint(L)
	new_base = np.random.randint(4)
	dE = EnergyChange(x[pos],new_base,pos)
	dx[np.random.random.randint(L),np.random.randint(3)] = 1

def next_array(x,L,J,h,N):
	for i in range(N):
		x = MonteCarloStep(x,L,J,h)
	return x

def num2string(x,L):
	seq = np.array(['A']*L)
	seq[x==1] = 'C'
	seq[x==2] = 'G'
	seq[x==3] = 'T'
	return ''.join(seq)

N = 100000
L = 40

nums = (np.random.randint(4,size=L) for x in range(N))
strings = (num2string(num,L) for num in nums)
seqs = (SeqRecord(Seq(string),id=f'sample {i}') for i,string in enumerate(strings))
SeqIO.write(seqs,'random_seqs.fasta','fasta')