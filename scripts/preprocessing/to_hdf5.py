import argparse
import h5py
from keras.utils import to_categorical
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class CSVreader():
    def __init__(self,filename,N,n_splits):
        self.filename = filename
        self.N = N
        self.n_splits = n_splits
        self.batch_n = self.N // self.n_splits
    def __len__(self):
        return self.n_splits
    def __getitem__(self,idx):
        if idx == self.n_splits - 1:
            data = pd.read_csv(self.filename,skiprows=(idx - 1) * self.batch_n,header=None).values
        else:
            data = pd.read_csv(self.filename,nrows=self.batch_n,skiprows=(idx - 1) * self.batch_n,header=None).values
        x = to_categorical(data[:,:-1])
        y = to_categorical(data[:,-1])
        return (x,y)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--infile',action='store',help='input file')
    parser.add_argument('-o','--outfile',action='store',help='output file')
    parser.add_argument('-n','--n_chunks',action='store',default=10,help='number of chunks')
    args = parser.parse_args()

    # Count lines in file & length of first line
    with open(args.infile,'r') as fh:
        line = fh.readline()
        L = len(line[:-3].split(','))
        N = sum(1 for line in fh) + 1
    
    # Create file reading object which loads data in chunks
    reader = CSVreader(args.infile,N,args.n_chunks)

    with h5py.File(args.outfile,'w') as outf:
        # Allocate memory
        outf.create_dataset('/x_train',shape=(n_train,L,4),dtype='i1')
        outf.create_dataset('/x_test',shape=(n_test,L,4),dtype='i1')
        outf.create_dataset('/y_train',shape=(n_train,),dtype='i1')
        outf.create_dataset('/y_test',shape=(n_test,),dtype='i1')

        # Track position in each array
        n_x_train = 0
        n_x_test = 0
        n_y_train = 0
        n_y_test = 0

        # Go through file and write
        for i in range(len(reader)):
            # Split chunk
            x_train,x_test,y_train,y_test = train_test_split(*reader[i])
            # Write to output
            outf['/x_train'][n_x_train:n_x_train + len(x_train),...] = x_train
            outf['/x_test'][n_x_test:n_x_test + len(x_test),...] = x_test
            outf['/y_train'][n_y_train:n_y_train + len(y_train)] = y_train
            outf['/y_test'][n_y_test:n_y_test + len(y_test)] = y_test
            # Advance position
            n_x_train += len(x_train)
            n_x_test += len(x_train)
            n_y_train += len(y_test)
            n_y_test += len(y_test)

if __name__ == '__main__':
    main()