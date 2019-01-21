import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-i1','--infile1',action='store',help='First input file (binding sequences)')
parser.add_argument('-i2','--infile2',action='store',help='First input file (control sequences)')
parser.add_argument('-o','--outfile',action='store',help='Output file')
args = parser.parse_args()

# Load binding sequences
x_bind = np.genfromtxt(args.infile1)
n_bind = x_bind.shape[0]
x_bind = x_bind.reshape(n_bind,40,3)

# Load control sequences
x_control = np.genfromtxt(args.infile2)
n_control = x_control.shape[0]
x_control = x_control.reshape(n_control,40,3)

# Combine, shuffle and split into training and test sets
x = np.concatenate((x_bind,x_control))
y = np.concatenate((np.ones(n_bind),np.zeros(n_control)))
ids = np.array(range(y.size))
ids.shuffle()
x = x[ids,...]
y = y[ids]
train_test = np.random.rand(y.size) > 0.8
x_train = x_train[np.logical_not(train_test),...]
y_train = y_train[np.logical_not(train_test)]
x_test = x_test[train_test,...]
y_test = y_test[train_test]

# Set up model and train
model = Sequential()
model.add(Conv1D(64, 3, activation='relu', input_shape=(40, 3)))
model.add(Conv1D(64, 3, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(128, 3, activation='relu'))
model.add(Conv1D(128, 3, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=50,
          batch_size=128)

# Test model
train_score = model.evaluate(x_train, y_train, batch_size=128)
test_score = model.evaluate(x_test, y_test, batch_size=128)
print(f'Training set loss = {score[0]:.2f}, accuracy = {score[1]:.2f}')
print(f'Test set loss = {score[0]:.2f}, accuracy = {score[1]:.2f}')

# Save model
model.save(args.outfile)
