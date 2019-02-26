import xgboost as xgb
import argparse
import numpy as np
import json
import pickle

# Take hyperparameter inputs
parser = argparse.ArgumentParser()
parser.add_argument('-i','--infile',action='store',help='Input file')
parser.add_argument('-o','--outfile',action='store',help='Output file tag')
parser.add_argument('-d','--max_depth',type=int, action='store',help='Tree depth')
parser.add_argument('-g','--gamma',type=float,action='store',help='Regularisation parameter')
parser.add_argument('-e','--eta',type=float,action='store',default=0.001,help='Learning rate')
parser.add_argument('-n','--n_round',type=int,action='store',default=10,help='Number of boosting rounds')
args = parser.parse_args()

# Import data
with open(args.infile,'rb') as fh:
    x_train,y_train,x_test,y_test = pickle.load(fh)
dtrain = xgb.Dmatrix(x_train,label=y_train)
dtest = xgb.Dmatrix(x_test,label=y_test)

# Train gradient booster
param = {'max_depth': args.max_depth, 'eta': args.eta, 'silent': 1, 'objective': 'binary:logistic'}
evallist = [(dtest, 'eval'), (dtrain, 'train')]
bst = xgb.train(param, dtrain, args.n_round, evallist)

# Test gradient booster
test_pred = np.stack((y_test,np.squeeze(bst.predict(dtest))))

# Save model
bst.save_model(args.outfile + '_bst.mdl')
# Save predictions on test set
np.savetxt(args.outfile + '_test_pred.txt.gz',test_pred)