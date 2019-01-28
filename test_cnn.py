from keras.models import load_model

model = load_model('models/cnn/cnn_3')

import numpy as np

x_train = np.load('data/IgE_train_features_stand.npy')
y_train = np.load('data/IgE_train_labels.npy')
x_test = np.load('data/IgE_test_features_stand.npy')
y_test = np.load('data/IgE_test_labels.npy')

# predict test labels
y_pred = np.squeeze(model.predict(x_test))
np.savetxt('models/cnn/cnn_3_test_pred.txt',np.stack((y_test,y_pred)).T)

from sklearn.metrics import roc_curve,roc_auc_score
auc = roc_auc_score(y_test,y_pred)
fpr,tpr,_= roc_curve(y_test,y_pred)
from matplotlib import pyplot as plt
plt.plot(fpr,tpr)
plt.fill_between(fpr,tpr,alpha=0.5)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title(f'ROC for classification using CNN 3\n AUC = {auc}')
plt.savefig('test_roc_cnn_3')
