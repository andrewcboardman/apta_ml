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
				file.write(','.join(record.astype(str).flatten())+'\n')
	elif args.mode == 'struct':
		strings = (str(seq.seq) for seq in seqs)
		coded_strings = (seqstruct2code(string) for string in strings)
		with open(args.outfile,'w') as file:
			for record in coded_strings:
				file.write(','.join(record.astype(str).flatten())+'\n')

if __name__ == '__main__':
	main()
