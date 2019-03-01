import argparse
import numpy as np
import os 
import pickle
import dask.array as da
import dask.dataframe as dd
from dask_ml.model_selection import train_test_split


parser = argparse.ArgumentParser()
parser.add_argument('-i1','--infile1',action='store',help='Input file 1')
parser.add_argument('-i2','--infile2',action='store',help='Input file 2')
parser.add_argument('-o','--outfile',action='store',help='Output file')
parser.add_argument('-s','--standardise',action='store_true',help='Standardise data')
args = parser.parse_args()

# load data from text files
data1 = dd.read_csv(args.infile1,sep=' ',header=None)
data2 = dd.read_csv(args.infile2,sep=' ',header=None)

# Add labels
data1['label'] = 0
data2['label'] = 1

# Concatenate
data1.append(data2)

# split to arrays
X = data1.values[:-1,:]
y = data1.values[-1,:]

if args.standardise:
	# Standardise features
	X_mean = da.mean(X,axis=0)
	X_std = da.std(X,axis=0)
	X_mean_arr = X_mean.compute()
	X_std_arr = X_std.compute()
	X_stand = (X - X_mean_arr) / X_std_arr

	# Split into training and test sets
	X_train, X_test, y_train, y_test = train_test_split(X_stand,y,train_size=0.8,test_size=0.2)
else:
	# Split into training and test sets
	X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8,test_size=0.2)

da.to_hdf5(args.outfile,{'/x_train':X_train,'/x_test':X_test,'/y_train':y_train,'/y_test':y_test})