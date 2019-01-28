import numpy as np
import time
t0 = time.time()
# Load binding sequences
x_bind = np.genfromtxt('data/plus_encode.txt')
n_bind = x_bind.shape[0]

# Load control sequences
x_control = np.genfromtxt('data/minus_encode.txt')
n_control = x_control.shape[0]

# Combine, shuffle and split into training and test sets
x = np.concatenate((x_bind,x_control))
y = np.concatenate((np.ones(n_bind),np.zeros(n_control)))
ids = np.array(range(y.size))

np.random.shuffle(ids)
x = x[ids,...]
y = y[ids]
train_test = np.random.rand(y.size) > 0.8


x_train = x[np.logical_not(train_test),...]
y_train = y[np.logical_not(train_test)]
x_test = x[train_test,...]
y_test = y[train_test]

from sklearn.naive_bayes import BernoulliNB
model = BernoulliNB()
model.fit(x_train,y_train)


from sklearn.metrics import roc_curve,roc_auc_score
y_pred = model.predict(x_test)
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
auc = roc_auc_score(y_test,y_pred)
print(auc)
print(time.time()-t0)