from keras.models import load_model

model = load_model('cnn_1')
model.summary()
import numpy as np

path = 'data/IgE'
x_test = np.load(f'{path}/test_features.npy')
y_test = np.load(f'{path}/test_labels.npy')

x_test = x_test.reshape(y_test.size,40,3)

# predict test labels
y_pred = np.squeeze(model.predict(x_test))
#np.savetxt('models/sk/cnn/cnn_1_test_pred.txt',np.stack((y_test,y_pred)).T)

from sklearn.metrics import roc_curve,roc_auc_score
auc = roc_auc_score(y_test,y_pred)
fpr,tpr,_= roc_curve(y_test,y_pred)
from matplotlib import pyplot as plt
plt.plot(fpr,tpr)
plt.fill_between(fpr,tpr,alpha=0.5)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title(f'ROC for classification of IgE data using deep neural network:\n AUC = {auc:.2f}')
plt.savefig('test_roc_deep_nn')
