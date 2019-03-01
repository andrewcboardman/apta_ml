import argparse
import numpy as np
import h5py
import dask.array as da
import dask.dataframe as dd
from dask_ml.linear_model import LogisticRegression
import pickle
# Take hyperparameter inputs
parser = argparse.ArgumentParser()
parser.add_argument('-i','--infile',action='store',help='Input file')
parser.add_argument('-o','--outfile',action='store',help='Output file tag')
parser.add_argument('-r','--reg',type=float,action='store',default=1.0,help='inverse regularisation strength')
args = parser.parse_args()

# Import data
f = h5py.File(args.infile)
x_train = da.from_array(f['/x_train'])
x_test = da.from_array(f['/x_test'])
y_train = da.from_array(f['/y_train'])
y_test = da.from_array(f['/y_test'])

# Train model
clf = LogisticRegression(C=args.reg)
clf.fit(x_train,y_train)

# Test model
y_predict = clf.predict(x_test)
test_pred = dd.DataFrame({'true':y_test,'pred':y_predict})
# Save predictions on test set
test_pred.to_csv(args.outfile + '_test_pred_*.csv')

# Save model 
with open(args.outfile + '_logit.pkl','wb') as fh:
	pickle.dump(clf, fh)

