import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

def select(samples, energies, threshold):
	return samples[energies<threshold]
def write_samples(samples,outfile):
	N,L,q = samples.shape
	seqs = np.array([['A','C','G','U'][x] for x in np.nonzero(high_affinity_samples)[2]]).reshape(N,L)
	SeqIO.write((SeqRecord(Seq(''.join(x))) for x in seqs),outfile,'fasta')

labels = np.load('/home/andrew/Downloads/samples_Ilk_pool_deep_E(1).npy')
samples = np.load('/home/andrew/Downloads/samples_Ilk_pool_deep(2).npy')

selected_samples = select(samples,energies,0.01)
write_samples(selected_samples,'MCMC_Samples.fa')
