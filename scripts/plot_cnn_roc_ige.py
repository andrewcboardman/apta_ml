import numpy as np 
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt


def rie(labels,predictions,alpha):
	"""Robust initial enrichment"""
	idx = np.flip(np.argsort(predictions))
	sorted_labels = labels[idx]
	ranks = np.arange(len(predictions))[np.squeeze(sorted_labels)==1] + 1
	top = np.sum(np.exp(-alpha * ranks / len(ranks)))/len(ranks)
	bottom = (1 - np.exp(-alpha)) * (len(predictions) * (np.exp(alpha / len(predictions)) - 1)) ** (-1)
	return top/bottom

def bedroc(labels,predictions,alpha):
	"""Boltzmann enhanced discrimination of the ROC"""
	rie_ = rie(labels,predictions,alpha)
	print(rie_)
	return rie_/alpha + (1 - np.exp(alpha))**(-1)


predictions = np.squeeze(np.load('/home/andrew/Downloads/IgE_deep_predictions.npy'))
labels = np.squeeze(np.load('/home/andrew/Downloads/IgE_deep_test.npy'))

fpr,tpr,_ = roc_curve(labels,predictions)

# plt.plot(fpr,tpr,color='red',label='Active vs inactive')
# plt.plot([0,1],[0,1],color='black',linestyle='dashed',label='Baseline')
# plt.legend()
# plt.xlabel('False positive rate')
# plt.ylabel('True positive rate')
# plt.show()
# #plt.savefig('figs/IgE_deep_cnn.png')


auroc = auc(fpr,tpr)
bedroc = bedroc(labels,predictions,0.01)
print(auroc)
print(bedroc)