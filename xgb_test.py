import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
bst = xgb.Booster(model_file='models/xgb/xgb_1')


x_train = np.load('data/IgE_train_features_stand.npy')
y_train = np.load('data/IgE_train_labels.npy')
x_test = np.load('data/IgE_test_features_stand.npy')
y_test = np.load('data/IgE_test_labels.npy')


dtest = xgb.DMatrix(x_test,label=y_test)
y_pred = bst.predict(dtest)
accuracy = np.mean(np.round(y_pred)==y_test)
print(accuracy)
from sklearn.metrics import roc_curve,roc_auc_score
auc = roc_auc_score(y_test,y_pred)
fpr,tpr,_= roc_curve(y_test,y_pred)
from matplotlib import pyplot as plt
plt.plot(fpr,tpr)
plt.fill_between(fpr,tpr,alpha=0.5)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title(f'ROC for classification using XGB 1\n AUC = {auc}')
plt.savefig('models/xgb/test_roc_xgb_1')