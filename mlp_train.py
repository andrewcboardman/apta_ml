import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

# Load binding sequences
x_bind = np.genfromtxt('plus_encode_stand.txt')
n_bind = x_bind.shape[0]

# Load control sequences
x_control = np.genfromtxt('minus_encode_stand.txt')
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


# In[13]:


model = Sequential()
model.add(Dense(64, input_dim=(120), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))


# In[14]:


model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=50,
          batch_size=128)


# In[22]:


train_score = model.evaluate(x_train, y_train, batch_size=128)
test_score = model.evaluate(x_test, y_test, batch_size=128)


# In[23]:


print(train_score,test_score)


# In[18]:


model2 = Sequential()
model2.add(Dense(80, input_dim=(120), activation='relu'))
model2.add(Dropout(0.3))
model2.add(Dense(80, activation='relu'))
model2.add(Dropout(0.3))
model2.add(Dense(80, activation='relu'))
model2.add(Dropout(0.3))
model2.add(Dense(1, activation='sigmoid'))


# In[20]:


model2.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model2.fit(x_train, y_train,
          epochs=50,
          batch_size=128)


# In[24]:


train_score2 = model2.evaluate(x_train, y_train, batch_size=128)
test_score2 = model2.evaluate(x_test, y_test, batch_size=128)


# In[25]:


print(train_score2,test_score2)


# In[26]:


model3 = Sequential()
model3.add(Dense(80, input_dim=(120), activation='relu'))
model3.add(Dropout(0.3))
model3.add(Dense(80, activation='relu'))
model3.add(Dropout(0.3))
model3.add(Dense(80, activation='relu'))
model3.add(Dropout(0.3))
model3.add(Dense(80, activation='relu'))
model3.add(Dropout(0.3))
model3.add(Dense(1, activation='sigmoid'))
model3.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model3.fit(x_train, y_train,
          epochs=50,
          batch_size=128)
train_score3 = model3.evaluate(x_train, y_train, batch_size=128)
test_score3 = model3.evaluate(x_test, y_test, batch_size=128)
print(train_score3,test_score3)


# In[27]:


model4 = Sequential()
model4.add(Dense(100, input_dim=(120), activation='relu'))
model4.add(Dropout(0.3))
model4.add(Dense(80, activation='relu'))
model4.add(Dropout(0.3))
model4.add(Dense(80, activation='relu'))
model4.add(Dropout(0.3))
model4.add(Dense(1, activation='sigmoid'))
model4.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model4.fit(x_train, y_train,
          epochs=50,
          batch_size=128)
train_score4 = model4.evaluate(x_train, y_train, batch_size=128)
test_score4 = model4.evaluate(x_test, y_test, batch_size=128)
print(train_score4,test_score4)


# In[36]:


from sklearn.metrics import roc_curve,roc_auc_score
y_pred2 = model2.predict(x_test)
fpr2, tpr2, thresholds2 = roc_curve(y_test, y_pred2)
auc2 = roc_auc_score(y_test,y_pred2)


# In[ ]:


import matplotlib.pyplot as plt
plt.plot(fpr2,tpr2)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC for model 2')
plt.savefig('Roc2.png')


# In[44]:


model5 = Sequential()
model5.add(Dense(150, input_dim=(120), activation='relu'))
model5.add(Dropout(0.5))
model5.add(Dense(150, activation='relu'))
model5.add(Dropout(0.5))
model5.add(Dense(150, activation='relu'))
model5.add(Dropout(0.5))
model5.add(Dense(100, activation='relu'))
model5.add(Dropout(0.5))
model5.add(Dense(100, activation='relu'))
model5.add(Dropout(0.5))
model5.add(Dense(1, activation='sigmoid'))
optimiser = Adam(lr=0.001)
model5.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model5.fit(x_train, y_train,
          epochs=50,
          batch_size=128)


# In[43]:


train_score5 = model5.evaluate(x_train, y_train, batch_size=128)
test_score5 = model5.evaluate(x_test, y_test, batch_size=128)
print(train_score5,test_score5)


# In[ ]:


model5 = Sequential()
model5.add(Dense(160, input_dim=(120), activation='relu'))
model5.add(Dropout(0.5))
model5.add(Dense(150, activation='relu'))
model5.add(Dropout(0.5))
model5.add(Dense(100, activation='relu'))
model5.add(Dropout(0.5))
model5.add(Dense(1, activation='sigmoid'))
optimiser = Adam(lr=0.001)
model5.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model5.fit(x_train, y_train,
          epochs=50,
          batch_size=128)

