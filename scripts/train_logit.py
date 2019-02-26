from sklearn.linear_model import LogisticRegression
import argparse
import numpy as np
import pickle

# Take hyperparameter inputs
parser = argparse.ArgumentParser()
parser.add_argument('-i','--infile',action='store',help='Input file')
parser.add_argument('-o','--outfile',action='store',help='Output file tag')
parser.add_argument('-r','--reg',type=float,action='store',help='inverse regularisation strength')
args = parser.parse_args()

# Import data
with open(args.infile,'rb') as fh:
    x_train,y_train,x_test,y_test = pickle.load(fh)

# Train model
clf = LogisticRegression(C=args.reg)
clf.fit(x_train,y_train)

# Test model
test_pred = np.stack((y_test,np.squeeze(clf.predict(x_test))))

# Save model 
with open(args.outfile + '_logit.pkl','wb') as fh:
	pickle.dump(clf, fh)
# Save predictions on test set
np.savetxt(args.outfile + '_test_pred.txt.gz',test_pred)
