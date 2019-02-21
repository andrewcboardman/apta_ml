import numpy as np

folder = 'sk'
# Load binding sequences
x_bind = np.genfromtxt(f'data/{folder}/bind_encode.txt')
n_bind = x_bind.shape[0]

# Load control sequences
x_control = np.genfromtxt(f'data/{folder}/control_encode.txt')
n_control = x_control.shape[0]

# Combine binding and control
x = np.concatenate((x_bind,x_control))
y = np.concatenate((np.ones(n_bind),np.zeros(n_control)))

ids = np.array(range(y.size))
np.random.shuffle(ids)
np.savetxt(f'data/{folder}/shuffle_ids.txt',ids.astype(int))
x = x[ids,...]
y = y[ids]

# Standardise
x_means = np.mean(x,axis=0)
x_stds = np.std(x,axis=0)
x_stand = (x - x_means) / x_stds

train_test = np.random.rand(y.size) > 0.8
np.savetxt(f'data/{folder}/train_test.txt',train_test.astype(bool))

x_train = x[np.logical_not(train_test),...]
x_train_stand = x_stand[np.logical_not(train_test),...]
y_train = y[np.logical_not(train_test)]
x_test = x[train_test,...]
x_test_stand = x_stand[train_test,...]
y_test = y[train_test]

np.save(f'data/{folder}/train_features',x_train)
np.save(f'data/{folder}/train_features_stand',x_train_stand)
np.save(f'data/{folder}/train_labels',y_train)
np.save(f'data/{folder}/test_features',x_test)
np.save(f'data/{folder}/test_features_stand',x_test_stand)
np.save(f'data/{folder}/test_labels',y_test)

