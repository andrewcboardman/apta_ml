import random
import argparse
import h5py
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-i','--infile',action='store',help='Input file')
parser.add_argument('-o','--outfile',action='store',help='Output file tag')
args = parser.parse_args()

# Count lines in file and size of first line
with open(args.infile,'r') as fh:
	L = len(fh.readline().split(','))
	n = sum(1 for line in fh) + 1

# Number of chunks to write file in 
chunksize = 10000
nchunks = n // chunksize 
last_chunk_size = n % chunksize

with h5py.File(args.outfile,'w') as output:
    # Create the structure
    output.create_dataset('/data',shape=(n,L))
    for i in range(nchunks):
        # Split csv file into chunks
        chunk = pd.read_csv(args.infile,header=None,skiprows=chunksize*i,nrows=chunksize,dtype=float)
        # Write to hdf 
        output['/data'][i*chunksize:(i+1)*chunksize] = chunk
    if last_chunk_size != 0:
        # Final chunk is a different size
        data = pd.read_csv(args.infile,skiprows=last_chunk_size,dtype=float)
        output['/data'][nchunks*chunksize:] = data.values
    # Shuffle data
    random.shuffle(output['/data'][...])
    # Separate into train and test features and labels
    output['/x_train'] = output['/data'][:int(n*0.8),:-1]
    output['/y_train'] = output['/data'][:int(n*0.8),-1]
    output['/x_test'] = output['/data'][int(n*0.8):,:-1]
    output['/y_test'] = output['/data'][int(n*0.8):,-1]
    # remove from dataset
    del output['/data']
