import argparse
import numpy as np
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dense, Dropout
from keras.optimizers import Adam
import json

# Take hyperparameter inputs
parser = argparse.ArgumentParser()
parser.add_argument('-i','--infile',action='store',help='Input file')
parser.add_argument('-o','--outfile',action='store',help='Output file tag')
parser.add_argument('--conv_n',type=int,action='store',default=50,help='Number of convolutional filters')
parser.add_argument('--conv_w',type=int,action='store',default=5,help='Width of convolutional units')
parser.add_argument('--dense_N',type=int,action='store',default=2,help='Number of dense layers')
parser.add_argument('--dense_n',type=int,action='store',default=100,help='Number of dense units per layer')
parser.add_argument('--dr',type=float,action='store',default=0.1,help='Dropout rate')
parser.add_argument('--lr',type=float,action='store',default=0.001,help='Learning rate for Adam optimizer')
parser.add_argument('-e','--epochs',type=int,action='store',default=10,help='Number of epochs to train for')
args = parser.parse_args()

# Import data
with open(args.infile,'rb') as fh:
    x_train,y_train,x_test,y_test = pickle.load(fh)

# Build network
model = Sequential()
model.add(Conv1D(args.conv_n, args.conv_w, activation='relu', input_shape=(40, 3)))
model.add(MaxPooling1D(3))
model.add(Flatten())
for i in range(args.dense_N):
    model.add(Dense(args.dense_n))
    model.add(Dropout(args.dr))
    model.add(Dense(args.dense_n))
    model.add(Dropout(args.dr))
model.add(Dense(1,activation='sigmoid'))
          
# Compile
history = model.compile(loss='binary_crossentropy',optimizer=Adam(lr=args.lr),metrics=['accuracy'])
          
# Train
model.fit(x_train,y_train,epochs=args.epochs,batch_size=128)

# Test
test_pred = np.stack((y_test,np.squeeze(model.predict(x_test))))
# Save model
model.save(args.outfile + '_cnn')
# Save model summary
with open(args.outfile + '_report.txt','w') as fh:
    # Pass the file handle in as a lambda function to make it callable
    model.summary(print_fn=lambda x: fh.write(x + '\n'))
# Save model history
with open(args.outfile + '_hist.json', 'w') as f:
    json.dump(hist.history, f)
# Save predictions on test set
np.savetxt(args.outfile + '_test_pred.txt.gz',test_pred)

          
          



