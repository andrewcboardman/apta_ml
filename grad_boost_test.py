import joblib

model = joblib.load('models/gradient_boosting/GBclf.joblib')

import numpy as np

# Load binding sequences
x_bind = np.genfromtxt('data/plus_encode_standardise.txt')
n_bind = x_bind.shape[0]

# Load control sequences
x_control = np.genfromtxt('data/minus_encode_standardise.txt')
n_control = x_control.shape[0]

# Combine binding and control
x = np.concatenate((x_bind,x_control))
y = np.concatenate((np.ones(n_bind),np.zeros(n_control)))

# load state of test-train split used for this model in training
split = np.genfromtxt('models/gradient_boosting/model_gb_split.txt')
ids = split[0,:].astype(int)
train_test = split[1,:].astype(bool)

# shuffle
x = x[ids,...]
y = y[ids]

# train-test split
x_train = x[np.logical_not(train_test),...]
y_train = y[np.logical_not(train_test)]
x_test = x[train_test,...]
y_test = y[train_test]


# predict test labels
y_pred = np.squeeze(model.predict(x_test))
np.savetxt('models/gradient_boosting/model_gb_test_pred.txt',np.stack((y_test,y_pred)).T)

from sklearn.metrics import roc_curve,roc_auc_score
auc = roc_auc_score(y_test,y_pred)
fpr,tpr,_= roc_curve(y_test,y_pred)
from matplotlib import pyplot as plt
plt.plot(fpr,tpr)
plt.fill_between(fpr,tpr,alpha=0.5)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title(f'ROC for classification using gradient_boosting\n AUC = {auc}')
plt.savefig('test_roc_model_gb')
