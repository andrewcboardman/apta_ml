import random
import argparse
import h5py
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('-i1','--infile1',action='store',help='Input file 1')
parser.add_argument('-i2','--infile2',action='store',help='Input file 2')
parser.add_argument('-o','--outfile',action='store',help='Output file tag')
parser.add_argument('-n',action='store',type=int,default=10000)
args = parser.parse_args()

x1 = pd.read_csv(args.infile1,header=None,dtype=float,sep=' ',nrows = args.n)
x2 = pd.read_csv(args.infile2,header=None,dtype=float,sep=' ',nrows = args.n)
y1 = np.zeros(args.n)
y2 = np.ones(args.n)
ix = np.arange(2*args.n)
np.random.shuffle(ix)

x = np.concatenate((x1,x2))[ix,:]
y = np.concatenate((y1,y2))[ix]

x = (x - np.mean(x,axis=0) ) / np.std(x,axis=0)

x_train, x_test, y_train, y_test = train_test_split(x,y)

with open(args.outfile,'wb') as fh:
    pickle.dump((x_train, x_test, y_train, y_test),fh)

