from dask_ml.xgboost import XGBClassifier
import argparse
import numpy as np
import json
import pickle
import h5py
import dask.array as da

# Take hyperparameter inputs
parser = argparse.ArgumentParser()
parser.add_argument('-i','--infile',action='store',help='Input file')
parser.add_argument('-o','--outfile',action='store',help='Output file tag')
parser.add_argument('-d','--max_depth',type=int, action='store',default=10,help='Tree depth')
parser.add_argument('-g','--gamma',type=float,action='store',default=0,help='Regularisation parameter')
parser.add_argument('-e','--eta',type=float,action='store',default=0.001,help='Learning rate')
parser.add_argument('-n','--n_round',type=int,action='store',default=10,help='Number of boosting rounds')
parser.add_argument('-c','--crossvalid',action='store_true',default=False,help='Perform 10-fold cross-validation')
args = parser.parse_args()

# Import data
f = h5py.File('test.hdf5','r')
x_train = da.from_array(f['/x_train'],chunks=1000)
x_test = da.from_array(f['/x_test'],chunks=1000)
y_train = da.from_array(f['/y_train'],chunks=1000)
y_test = da.from_array(f['/y_test'],chunks=1000)
x_train = da.concatenate((x_train[:,:,0],
	x_train[:,:,1],
	x_train[:,:,2],
	x_train[:,:,3]), axis=1)
x_test = da.concatenate((x_test[:,:,0],
	x_test[:,:,1],
	x_test[:,:,2],
	x_test[:,:,3]), axis=1)


print(y_train.reshape(80,1))

# Set hyperparameters
#param = {'max_depth': args.max_depth, 'eta': args.eta, 'silent': 1, 'objective': 'binary:logistic'}
params = {'objective': 'binary:logistic',
          'max_depth': 4, 'eta': 0.01, 'subsample': 0.5,
          'min_child_weight': 0.5}
from dask.distributed import Client
client = Client()
import dask_xgboost
bst = dask_xgboost.train(client, params, x_train, y_train, num_boost_round=10)
assert 1==0

if args.crossvalid:
	dtrain = xgb.DMatrix(x_train,labels=y_train)
	history = xgb.cv(param,dtrain,num_boost_round=args.n_round, nfold=10,
             metrics={'error'}, seed=0,
             callbacks=[xgb.callback.print_evaluation(show_stdv=False),xgb.callback.early_stop(3)])

	# save cross validation history
	history.to_csv(args.outfile + '_history.csv')

else:
	# Train gradient booster
	clf = XGBClassifier(max_depth= args.max_depth, learning_rate= args.eta)
	clf.fit(x_train,y_train)

	# Test gradient booster
	y_predict = clf.predict(x_test)
	test_pred = dd.DataFrame({'true':y_test,'pred':y_predict})

	# Save predictions on test set
	test_pred.to_csv(args.outfile + '_test_pred_*.csv')

	# Save model
	clf.save(args.outfile + '_model')
	