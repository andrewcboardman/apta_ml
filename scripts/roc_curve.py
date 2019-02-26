import argparse
from sklearn.metrics import roc_curve,roc_auc_score
from matplotlib import pyplot as plt
import numpy
# Take inputs
parser = argparse.ArgumentParser()
parser.add_argument('-i','--infile',action='store',help='Input file')
parser.add_argument('-o','--outfile',action='store',help='Output file tag')

test_pred = np.genfromtxt(args.infile)
y_test = test_pred[0,:]
y_pred = test_pred[1,:]

auc = roc_auc_score(y_test,y_pred)
fpr,tpr,_= roc_curve(y_test,y_pred)

plt.plot(fpr,tpr)
plt.fill_between(fpr,tpr,alpha=0.5)
plt.xlabel('FPR')
plt.ylabel('TPR')
title = input("What should the title be?")
plt.title(title + f'\n AUC = {auc:.2f}')
plt.savefig('test_roc_deep_nn')