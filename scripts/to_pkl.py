import argparse
import h5py
from keras.utils import to_categorical
import numpy as np

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-i','--infile',action='store',help='input file')
	parser.add_argument('-o','--outfile',action='store',help='output file')
	args = parser.parse_args()

	# count lines in file
	with open(args.infile,'r') as fh:
		line = fh.readline()
		L = len(line[:-3].split(','))
		N = sum(1 for line in fh) + 1

	# Train-test split
	n_train = int(np.round(0.8*N))
	n_test = N - n_train

	
	with open(args.infile,'r') as inf:
		# Allocate memory
		s = np.zeros((N,),dtype='U40')
		y = np.zeros((N,L))
		# Go through file and write
		for i,line in enumerate(inf):
			y[i] = int(line[-2])
			s[i] = np.array(line[:-3].split(','),dtype='int8')
		x = to_categorical(s,num_classes=4,dtype='int8')
		x_train=np.zeros((n_train,L,4),dtype='int8')
		x_test=np.zeros((n_test,L,4),dtype='int8')
		y_train=np.zeros((n_train,),dtype='int8')
		y_test=np.zeros((n_test,),dtype='int8')
if __name__ == '__main__':
	main()