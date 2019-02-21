import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout,Reshape
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras import A

path = 'data/IgE'
x_train = np.load(f'{path}/train_features_stand.npy')
y_train = np.load(f'{path}/train_labels.npy')
x_test = np.load(f'{path}/test_features_stand.npy')
y_test = np.load(f'{path}/test_labels.npy')

n_train = x_train.shape[0]
x_train = x_train.reshape(n_train,40,3)
n_test = x_test.shape[0]
x_test = x_test.reshape(n_test,40,3)
print(y_test.shape)

# Set up model and train
model = Sequential()
model.add(Conv1D(30, 5, activation='relu', input_shape=(40, 3)))
model.add(MaxPooling1D(3))
model.add(Reshape((360,)))
model.add(Dense(100))
model.add(Dropout(0.1))
model.add(Dense(100))
model.add(Dropout(0.1))
model.add(Dense(100))
model.add(Dropout(0.1))
model.add(Dense(100))
model.add(Dropout(0.1))
model.add(Dense(100))
model.add(Dropout(0.1))
model.add(Dense(100))
model.add(Dense(1, activation='sigmoid'))

model.summary()


model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=10,
          batch_size=128)


# Save model
model.save('cnn_1')