import numpy as np 
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

predictions = np.load('/home/andrew/Downloads/Ilk_Pool_deep_predictions.npy')
labels = np.load('/home/andrew/Downloads/Ilk_Pool_deep_test.npy')

fpr,tpr,_ = roc_curve(labels,predictions)

plt.plot(fpr,tpr,color='red',label='Active vs inactive')
plt.plot([0,1],[0,1],color='black',linestyle='dashed',label='Baseline')
plt.legend()
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.savefig('figs/Ilk_pool_deep_cnn_roc.png')