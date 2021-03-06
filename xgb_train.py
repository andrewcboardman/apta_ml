import xgboost as xgb
import argparse
import numpy as np
import json
import pickle

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
with open(args.infile,'rb') as fh:
    x_train,y_train,x_test,y_test = pickle.load(fh)
dtrain = xgb.DMatrix(x_train,label=y_train)
dtest = xgb.DMatrix(x_test,label=y_test)

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
	evallist = [(dtest, 'eval'), (dtrain, 'train')]
	bst = xgb.train(param, dtrain, args.n_round, evallist)

	# Test gradient booster
	test_pred = np.stack((y_test,np.squeeze(bst.predict(dtest))))

	# Save model
	bst.save_model(args.outfile + '_bst.mdl')
	# Save predictions on test set
	np.savetxt(args.outfile + '_test_pred.txt.gz',test_pred)