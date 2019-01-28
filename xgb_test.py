import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
bst = xgb.Booster(model_file='models/xgb/xgb_1')
# xgb.plot_tree(bst)
# plt.savefig('tree',dpi=400)
# # Load binding sequences
x_bind = np.genfromtxt('data/plus_test.txt')
n_bind = x_bind.shape[0]

# Load control sequences
x_control = np.genfromtxt('data/minus_test.txt')
n_control = x_control.shape[0]

# Combine, shuffle and split into training and test sets
x = np.concatenate((x_bind,x_control))
y = np.concatenate((np.ones(n_bind),np.zeros(n_control)))
ids = np.array(range(y.size))

np.random.shuffle(ids)
x = x[ids,...]
y = y[ids]
train_test = np.random.rand(y.size) > 0
# train-test split
x_train = x[np.logical_not(train_test),...]
y_train = y[np.logical_not(train_test)]
x_test = x[train_test,...]
y_test = y[train_test]


 #plt.show()

# dtest = xgb.DMatrix(x_test,label=y_test)
# preds = bst.predict(dtest)
# accuracy = np.mean(np.round(preds)==y_test)
# print(accuracy)
