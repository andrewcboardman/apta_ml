from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import numpy as np

def IsingEnergy(x,h,J,L):
	"""Finds the Ising energy for a sample sequence"""
	field_energy = -np.sum(h[range(L),x])
	coupling_energy = -np.sum(J[range(L),x,range(L),x])
	return field_energy + coupling_energy

def EnergyChange(x,new_base,pos,J,h,L):
	# Field energy for new base - field energy for new base
	
	dE = - h[pos,x.astype(int)[pos]] + h[pos,new_base]
	# coupling energy for new base
	dE = dE - np.sum(J[range(L),x,pos,x[pos]]) + np.sum(J[range(L),x,pos,new_base])
	return dE

def MonteCarloStep(x,L,J,h):
	pos = np.random.randint(L)
	new_base = np.random.randint(4)
	dE = EnergyChange(x,new_base,pos,J,h,L)
	if np.exp(-dE) > np.random.rand():
		x[pos] = new_base
	return x


def next_array(x,L,J,h,n):
	for i in range(n):
		x = MonteCarloStep(x,L,J,h)
	return x

def num2string(x,L):
	seq = np.array(['A']*L)
	seq[x==1] = 'C'
	seq[x==2] = 'G'
	seq[x==3] = 'T'
	return ''.join(seq)

def main():
	N = 100 # number of samples to generate
	L = 40 # number of spins
	q = 4 # number of states
	n = 10 # number of Monte Carlo steps between samples

	# Initialise simulation and save coefficients
	h = np.random.rand(L,q)/np.sqrt(120)
	np.savetxt('h.txt',h)
	J = np.random.rand(L,q,L,q)/np.sqrt(120)
	J = 0.5*(J+J.transpose(2,3,0,1)) # symmetrise J
	nums = np.zeros((N,L)).astype(int)
	energy = np.zeros(N)
	nums[0,...] = np.random.randint(q,size=L)
	for i in range(1,N):
		nums[i,...] = next_array(nums[i-1,...],L,J,h,n)
		energy[i] = IsingEnergy(nums[i,...],h,J,L)
	strings = (num2string(num,L) for num in nums)
	seqs = (SeqRecord(Seq(string),id=f'sample {i}') for i,string in enumerate(strings))
	SeqIO.write(seqs,'test_seqs.fasta','fasta')

if __name__ == '__main__':
	main()