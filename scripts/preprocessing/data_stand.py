import argparse
import dask.array as da
import dask.dataframe as dd

parser = argparse.ArgumentParser()
parser.add_argument('-i1','--infile1',action='store',help='Input file 1')
parser.add_argument('-i2','--infile2',action='store',help='Input file 2')
parser.add_argument('-o','--outfile',action='store',help='Output path')
parser.add_argument('-s','--standardise',action='store_true',help='Standardise data')
args = parser.parse_args()

# load data from text files
data1 = dd.read_csv(args.infile1,sep=' ',header=None)
data2 = dd.read_csv(args.infile2,sep=' ',header=None)

# Add class labels
data1['label'] = 0
data2['label'] = 1

# Concatenate
data1.append(data2)

# split to arrays
X = data1.drop('label',axis=1).values
y = data1['label']

# Standardise features
X_mean = da.mean(X,axis=0)
X_std = da.std(X,axis=0)
X_mean_arr = X_mean.compute()
X_std_arr = X_std.compute()
data_stand = (data1.drop('label',axis=1) - X_mean_arr) / X_std_arr
data_stand['label'] = y
data_stand.to_csv(args.outfile+'r*_.csv',header=None,index=False)