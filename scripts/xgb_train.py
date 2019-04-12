import argparse
import numpy as np
import json
import pickle
import h5py
import xgboost as xgb
import pandas as pd

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
x_train = np.array(f['/x_train'])
x_test = np.array(f['/x_test'])
y_train = np.array(f['/y_train'])
y_test = np.array(f['/y_test'])
x_train = x_train.reshape(f['/n_train'][...],160)
x_test = x_test.reshape(f['/n_test'][...],160)



# Set hyperparameters
params = {'max_depth': args.max_depth, 'eta': args.eta, 'silent': 1, 'objective': 'binary:logistic'}

dtrain = xgb.DMatrix(x_train,label=y_train)
dtest = xgb.DMatrix(x_test,label=y_test)


if not args.crossvalid:
	# Train gradient booster
	clf = xgb.train(params,dtrain)

	# Test gradient booster
	y_predict = clf.predict(dtest)
	test_pred = pd.DataFrame({'true':y_test,'pred':y_predict})

	# Save predictions on test set
	test_pred.to_csv(args.outfile + '_test_pred.csv')

	# Save model
	with open(args.outfile + '_model','wb') as fh:
		pickle.dump(clf,fh)

else:

	history = xgb.cv(param,dtrain,num_boost_round=args.n_round, nfold=10,
             metrics={'error'}, seed=0,
             callbacks=[xgb.callback.print_evaluation(show_stdv=False),xgb.callback.early_stop(3)])

	# save cross validation history
	history.to_csv(args.outfile + '_history.csv')
	
	