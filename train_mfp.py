import numpy as np
from mf_potts import MFPottsClassifier
import joblib
x_train = np.load('data/IgE_train_features.npy')
n_train = x_train.shape[0]
x_train = x_train.reshape(n_train,40,3)
y_train = np.load('data/IgE_train_labels.npy')
x_test = np.load('data/IgE_test_features.npy')
n_test = x_test.shape[0]
x_test = x_test.reshape(n_test,40,3)
y_test = np.load('data/IgE_test_labels.npy')

model = MFPottsClassifier()
model.fit(x_train,y_train)
model.save('mfp_2.joblib')

# model = joblib.load('mfp.joblib')
y_pred = model.predict(x_test)
np.savetxt('mfp_2_logit_test_pred.txt',np.stack((y_test,y_pred)).T)

from sklearn.metrics import roc_curve,roc_auc_score
auc = roc_auc_score(y_test,y_pred)
fpr,tpr,_= roc_curve(y_test,y_pred)
from matplotlib import pyplot as plt
plt.plot(fpr,tpr)
plt.fill_between(fpr,tpr,alpha=0.5)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title(f'ROC for classification using mean-field potts\n AUC = {auc}')
plt.savefig('test_roc_mfp_2')




