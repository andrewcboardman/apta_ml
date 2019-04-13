import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve,auc

def calc_bedroc(y,x):
	data = pd.DataFrame({'x':x,'y':y})
	data.sortby('x',inplace=True)
	n = len(data)
	

data = pd.read_csv('~/Downloads/Ilk_pool_mn_E.txt',header = None,sep = ' ')
data.columns = ['Pool 5','Random','Pool 2']

x0 = -np.concatenate((data['Pool 5'].values,data['Pool 2'].values))
y0 = np.concatenate((np.ones_like(data['Pool 5'].values),np.zeros_like(data['Pool 2'].values)))

fpr0,tpr0,_ = roc_curve(y0,x0)

auroc = auc(fpr0,tpr0)
# bedroc = 

plt.plot(fpr0,tpr0,color='red',label='Pool 5 vs Pool 2')

plt.plot([0,1],[0,1],color='black',linestyle='dashed',label='Baseline')

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.legend()
plt.savefig('figs/mn_ROC.png')