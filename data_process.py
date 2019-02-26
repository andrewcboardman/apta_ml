import argparse
import numpy as np
import os 
import pickle
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser()
parser.add_argument('-i1','--infile1',action='store',help='Input file 1')
parser.add_argument('-i2','--infile2',action='store',help='Input file 2')
parser.add_argument('-o','--outfile',action='store',help='Output file')
args = parser.parse_args()

# load data from text files
data1 = np.genfromtxt(args.infile1)
n1 = data1.shape[0]
data2 = np.genfromtxt(args.infile2)
n2 = data2.shape[0]
    
# Make inputs into 1 array
features = np.concatenate((data1,data2))
labels = np.concatenate((np.zeros(n1),np.ones(n2)))

# Standardise features
features_stand = (features - np.mean(features,axis=0)) / (np.std(features,axis=0) + 0.01)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features,labels,train_size=0.8,test_size=0.2)


# Pickle numpy arrays
with open(args.outfile,'wb') as fid:
    pickle.dump((X_train,y_train,X_test,y_test),fid)
