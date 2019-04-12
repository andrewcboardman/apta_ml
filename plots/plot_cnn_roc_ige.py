import numpy as np 
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt


predictions = np.squeeze(np.load('/home/andrew/Downloads/IgE_shallow_predictions.npy'))
labels = np.squeeze(np.load('/home/andrew/Downloads/IgE_shallow_test.npy'))

fpr,tpr,_ = roc_curve(labels,predictions)

plt.plot(fpr,tpr,color='red',label='Active vs inactive')
plt.plot([0,1],[0,1],color='black',linestyle='dashed',label='Baseline')
plt.legend()
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.savefig('figs/IgE_shallow_cnn_roc.png')


auroc = auc(fpr,tpr)
