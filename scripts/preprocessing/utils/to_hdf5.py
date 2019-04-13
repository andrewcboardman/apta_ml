import argparse
import h5py
from keras.utils import to_categorical
import numpy as np
from Bio import SeqIO

def encode(rec,mydict):
	seq = str(rec.seq)
	ix = str(rec.id)
	chars = np.array(list(seq))
	code = np.vectorize(mydict.__getitem__)(chars)
	label = int(ix[-1])
	return (to_categorical(code,num_classes=4,dtype='int8'),label)

def main():
	""" Converts a FASTA file into a HDF archive"""
	parser = argparse.ArgumentParser()
	parser.add_argument('-i','--infile',action='store',help='input file')
	parser.add_argument('-o','--outfile',action='store',help='output file')
	args = parser.parse_args()

	base_dict = {'A':0,'C':1,'G':2,'T':3}

	# load data from FASTA files
	seqs = SeqIO.parse(args.infile,'fasta')

	# count length of sequences
	first_seq = next(seqs)
	L = len(first_seq.seq)
	
	# count lines in file
	N = sum(1 for seq in seqs) + 1

	# Reload data
	seqs = SeqIO.parse(args.infile,'fasta')

	# Translate to 
	seqs_encode = (encode(rec,base_dict) for rec in seqs)
	# Train-test split
	n_train = int(np.round(0.8*N))
	n_test = N - n_train

	
	with open(args.infile,'r') as inf, h5py.File(args.outfile,'w') as outf:
		# Allocate memory
		outf.create_dataset('/x_train',shape=(n_train,L,4),dtype='i1')
		outf.create_dataset('/x_test',shape=(n_test,L,4),dtype='i1')
		outf.create_dataset('/y_train',shape=(n_train,),dtype='i1')
		outf.create_dataset('/y_test',shape=(n_test,),dtype='i1')
		# Go through file and write
		for i,(x,y) in enumerate(seqs_encode):
			if i < n_train:
				outf['/x_train'][i,...] = x
				outf['/y_train'][i] = y
			else:
				outf['/x_test'][i-n_train,...] = x
				outf['/y_test'][i-n_train] = y

if __name__ == '__main__':
	main()