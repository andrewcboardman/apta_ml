import argparse
import numpy as np
import os 
import pickle
import pandas as pd
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import h5py
from keras.utils import to_categorical

dicto = {'A':0,'C':1,'G':2,'T':3}
def seq2code(seq):
	seq_arr = list(str(seq)[2:-1])
	class_arr = np.array([dicto[x] for x in seq_arr])
	return to_categorical(class_arr,num_classes=4,dtype='int8')


parser = argparse.ArgumentParser()
parser.add_argument('-i1','--infile1',action='store',help='Binding sequences')
parser.add_argument('-i0','--infile0',action='store',help='Control sequences')
parser.add_argument('-o','--outfile',action='store',help='Output file')
parser.add_argument('-s','--standardise',action='store_true',help='Standardise data')
args = parser.parse_args()

# load data from FASTA files for counting
seqs1 = SeqIO.parse(args.infile1,'fasta')
seqs0 = SeqIO.parse(args.infile0,'fasta')

# Count
n1 = sum((1 for seq in seqs1))
n0 = sum((1 for seq in seqs0))

# reload data 
seqs1 = SeqIO.parse(args.infile1,'fasta')
seqs0 = SeqIO.parse(args.infile0,'fasta')

# write to HDF file
with h5py.File(args.outfile,'w') as output:

	output['/n1'] = n1	
	output.create_dataset('/seqs1',shape=(n1,),dtype='S40')
	for i,seq in enumerate(seqs1):
		output['/seqs1'][i] = np.array(str(seq.seq),dtype='S40')
	output['/n0'] = n0
	output.create_dataset('/seqs0',shape=(n0,),dtype='S40')
	for j,seq in enumerate(seqs0):
		output['/seqs0'][j] = np.array(str(seq.seq),dtype='S40')
	# Shuffle and split into training and test datasets
	N = n0 + n1
	n_train = np.round(0.8*N).astype(int)
	n_test = N - n_train
	output['/n_train'] = n_train
	output['/n_test'] = n_test

	mix = np.concatenate((np.ones(n1),np.zeros(n0)))
	ix = np.concatenate((np.arange(n1),np.arange(n0)))
	split = np.stack((mix,ix),axis=-1)
	np.random.shuffle(split)
	train_split = split[:n_train]
	test_split = split[n_train:]

	output.create_dataset('/s_train',shape=(n_train,),dtype='S40')
	output.create_dataset('/y_train',shape=(n_train,),dtype='i8')
	output.create_dataset('/s_test',shape=(n_test,),dtype='S40')
	output.create_dataset('/y_test',shape=(n_test,),dtype='i8')
	# Write shuffled sequences and labels to file
	for k, sp in enumerate(train_split):
		if sp[0] == 0:
			output['/s_train'][k] = output['/seqs0'][sp[1]]
			output['/y_train'][k] = 0
		else:
			output['/s_train'][k] = output['/seqs1'][sp[1]]
			output['/y_train'][k] = 1
	for k, sp in enumerate(test_split):
		if sp[0] == 0:
			output['/s_test'][k] = output['/seqs0'][sp[1]]
			output['/y_test'][k] = 0
		else:
			output['/s_test'][k] = output['/seqs1'][sp[1]]
			output['/y_test'][k] = 0
	del output['/seqs0']
	del output['/seqs1']

	# Encode shuffled sequences
	output.create_dataset('/x_train',shape=(n_train,40,4),dtype='i1')
	output.create_dataset('/x_test',shape=(n_test,40,4),dtype='i1')
	for i,seq in enumerate(output['/s_train']):
		output['/x_train'][i,:,:] = seq2code(output['/s_train'][i])
	for i,seq in enumerate(output['/s_test']):
		output['/x_test'][i,:,:] = seq2code(output['/s_test'][i])
	output['/s_train']
	output['/s_test']