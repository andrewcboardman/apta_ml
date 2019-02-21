import numpy as np
import argparse
import os
from Bio import SeqIO
import itertools


def seq2code(seq):
	seq_arr = np.array(list(seq))[np.newaxis,:]
	code_arr = np.squeeze(np.stack(3*[np.zeros_like(seq_arr,dtype=int)]))
	code_arr[seq_arr==np.array(('C','G','T'))[:,np.newaxis]] = 1
	return code_arr.T
def seqstruct2code(seq):
	seq_arr = np.array(list(seq))
	seq_arr = np.core.defchararray.add(seq_arr[:len(seq_arr)//2],seq_arr[len(seq_arr)//2:])
	code_arr = np.squeeze(np.stack(24*[np.zeros_like(seq_arr,dtype=int)]))
	seq_alphabet = ['A','C','G','T']
	struct_alphabet = ['F','H','I','M','S','T']
	full_alphabet = np.array([x+y for (x,y) in itertools.product(seq_alphabet,struct_alphabet)])
	code_arr[seq_arr==full_alphabet[:,np.newaxis]] = 1
	return code_arr.T
# def seq2array_indels(seq,n_spins):
# 	array = np.zeros((n_spins,4),dtype=int)
# 	seqarr = np.array(list(seq))[:,np.newaxis]
# 	array[seqarr==np.array(('A','C','G','T'))] = 1
# 	return array

# def seq2array_ortho(seq,n_spins):
# 	array = np.zeros((n_spins,3),dtype=int)
# 	for i in range(n_spins):
# 		if seq[i] == 'A':
# 			array[i,:] = (-3,0,0)
# 		elif seq[i] == 'C':
# 			array[i,:] = (1,-2,0)
# 		elif seq[i] == 'G':
# 			array[i,:] = (1,1,-1)
# 		elif seq[i] == 'T':
# 			array[i,:] = (1,1,1)
# 	return array



def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--infile', type=str, action='store', dest='infile',	help='FASTA-formatted samples')
	parser.add_argument('-o', '--outfile', type=str, action='store', dest='outfile', help='name of encoded text file')
	parser.add_argument('-m', '--mode', type=str, action='store', dest='mode', default='bases',help='Method of encoding')
	args = parser.parse_args()

	# Read FASTA-formatted samples
	seqs = SeqIO.parse(args.infile,'fasta')


	if args.mode == 'bases':
		strings = (str(seq.seq) for seq in seqs)
		coded_strings = (seq2code(string) for string in strings)
		with open(args.outfile,'w') as file:
			for record in coded_strings:
				file.write(' '.join(record.astype(str).flatten())+'\n')
	elif args.mode == 'struct':
		strings = (str(seq.seq) for seq in seqs)
		coded_strings = (seqstruct2code(string) for string in strings)
		with open(args.outfile,'w') as file:
			for record in coded_strings:
				file.write(' '.join(record.astype(str).flatten())+'\n')
	# elif args.mode == 'indels':
	# 	strings = (str(seq.seq) for seq in seqs)
	# 	coded_strings = (seq2array_indels(string,n_spins) for string in strings)
	# 	with open(args.path + args.outfile,'w') as file:
	# 		for record in coded_strings:
	# 			file.write(' '.join(record[:,0].astype(str)) + '\n' + \
	# 				' '.join(record[:,1].astype(str)) + '\n' + \
	# 				' '.join(record[:,2].astype(str)) + '\n' + \
	# 				' '.join(record[:,3].astype(str)) + '\n')

	# elif args.mode == 'ortho':
	# 	strings = (str(seq.seq) for seq in seqs)
	# 	coded_strings = (seq2array_ortho(string,n_spins) for string in strings)
	# 	with open(args.path + args.outfile,'w') as file:
	# 		for record in coded_strings:
	# 			file.write(' '.join(record[:,0].astype(str)) + '\n' + \
	# 				' '.join(record[:,1].astype(str)) + '\n' + \
	# 				' '.join(record[:,2].astype(str)) + '\n')

if __name__ == '__main__':
	main()
