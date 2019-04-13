from dask_ml.xgboost import XGBClassifier
import argparse
import numpy as np
import json
import pickle
import h5py

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
f = h5py.File(args.infile)
x_train = da.from_array(f['/x_train'])
x_test = da.from_array(f['/x_test'])
y_train = da.from_array(f['/y_train'])
y_test = da.from_array(f['/y_test'])

# Set hyperparameters
param = {'max_depth': args.max_depth, 'eta': args.eta, 'silent': 1, 'objective': 'binary:logistic'}


if args.crossvalid:
	history = xgb.cv(param,dtrain,num_boost_round=args.n_round, nfold=10,
             metrics={'error'}, seed=0,
             callbacks=[xgb.callback.print_evaluation(show_stdv=False),xgb.callback.early_stop(3)])

	# save cross validation history
	history.to_csv(args.outfile + '_history.csv')

else:
    # Train gradient booster
    clf = XGBClassifier(max_depth= args.max_depth, learning_rate= args.eta,n_estimators = args.n_round)
    clf.fit(x_train,y_train)
    
    # Test gradient booster
    y_predict = clf.predict(x_test)

    # Save predictions on test set
    with h5py.File(args.outfile,'w') as f:
        f.create_dataset('/true',data=y_test)
        f.create_dataset('/pred',data=y_predict)
    # Save model
    clf.save(args.outfile + '_model')
