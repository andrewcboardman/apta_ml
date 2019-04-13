from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import numpy as np
from scipy.signal import correlate
import matplotlib.pyplot as plt

def IsingEnergy(seq,h,J,N):
	"""Finds the Ising energy for each sample sequence"""
	field_energy = -np.tensordot(h,seq)
	coupling_energy = -np.tensordot(seq,np.tensordot(seq,J,axes=((0,1),(2,3))))
	return field_energy + coupling_energy

def next_array(seq,L,J,h,n):
	x = np.copy(seq)
	for i in range(n):
		x = GibbsSample(x,L,J,h)
	return x

def GibbsSample(seq,L,J,h):
	# Propose changes to sequence
	which_pos = np.random.randint(L)
	old_base = seq[which_pos,:]
	new_base = (np.random.randint(4)==np.arange(1,4)).astype(int)
	base_change = new_base - old_base

	# Calculate changes in energy using Ising model
	field_energy_change = np.sum(h[which_pos,:] * base_change)
	coupling_energy_change = -np.sum(np.sum(J[which_pos,...] * base_change[...,np.newaxis,np.newaxis], axis=0) * seq) 
	delta = field_energy_change + coupling_energy_change

	# Accept or reject changes based on energy change
	new_seq = np.copy(seq)
	if np.exp(-delta)>np.random.rand():
		new_seq[which_pos,:] += base_change
	return new_seq

def num2string(x,L):
	seq = np.array(['A']*L)
	seq[x[:,0].astype(bool)] = 'C'
	seq[x[:,1].astype(bool)] = 'G'
	seq[x[:,2].astype(bool)] = 'T'
	return ''.join(seq)

def main():
	N = 1000000 # number of samples to generate
	L = 40 # number of spins
	q = 3 # number of states
	n = 200 # number of Monte Carlo steps between samples
	Nb = 1000 # number of burn in steps
	b = 1 # inverse temperature

	# Initialise simulation 
	h = b*np.genfromtxt('data/sk/h_sk.txt').reshape(L,q)
	J = b*np.genfromtxt('data/sk/J_sk.txt').reshape(L,q,L,q)
	nums = np.zeros((N,L,q)).astype(int)
	energy = np.zeros(N)
	nums[0,range(L),np.random.randint(q,size=L)] = 1

	# Burn in simulation for Nb steps
	for i in range(Nb):
		nums[0,...] = next_array(nums[0,...],L,J,h,n)
	# Run simulation for N steps
	for i in range(1,N):
		nums[i,...] = next_array(nums[i-1,...],L,J,h,n)
		energy[i] = IsingEnergy(nums[i,...],h,J,L)

	# convert to sequence format and write to file
	strings = (num2string(num,L) for num in nums)
	seqs = (SeqRecord(Seq(string),id=f'sample {i}') for i,string in enumerate(strings))
	SeqIO.write(seqs,'data/sk/sk2_seqs.fasta','fasta')
	np.savetxt('data/sk/sk2_energies.txt',energy)

if __name__ == '__main__':
	main()