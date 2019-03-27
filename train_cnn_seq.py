import argparse
import numpy as np
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dense, Dropout,Flatten,GlobalAveragePooling1D
from keras.optimizers import Adam
import json
from sklearn.model_selection import StratifiedKFold
import pickle
import h5py
from keras.utils.io_utils import HDF5Matrix
import pandas as pd


# Take hyperparameter inputs
parser = argparse.ArgumentParser()
parser.add_argument('-i','--infile',action='store',help='Input file')
parser.add_argument('-o','--outfile',action='store',help='Output file tag')
parser.add_argument('--conv_n',type=int,action='store',default=50,help='Number of convolutional filters')
parser.add_argument('--conv_w',type=int,action='store',default=5,help='Width of convolutional units')
parser.add_argument('--dr',type=float,action='store',default=0.1,help='Dropout rate')
parser.add_argument('--lr',type=float,action='store',default=0.001,help='Learning rate for Adam optimizer')
parser.add_argument('-e','--epochs',type=int,action='store',default=10,help='Number of epochs to train for')
parser.add_argument('-c','--crossvalid',action='store_true',default=False,help='Perform 10-fold cross-validation')
args = parser.parse_args()

# Import data
x_train = HDF5Matrix(args.infile,'/x_train')
x_test = HDF5Matrix(args.infile,'/x_test')
y_train = HDF5Matrix(args.infile,'/y_train')
y_test = HDF5Matrix(args.infile,'/y_test')
print(y_train.shape)
# Build network
model = Sequential()
model.add(Conv1D(args.conv_n, args.conv_w, activation='relu', input_shape=(40, 4)))
model.add(Conv1D(args.conv_n, args.conv_w, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Dropout(args.dr))
model.add(Conv1D(2*args.conv_n, args.conv_w, activation='relu'))
model.add(Conv1D(2*args.conv_n, args.conv_w, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dropout(args.dr))
model.add(Dense(1,activation='sigmoid'))
          
# Compile
model.compile(loss='binary_crossentropy',optimizer=Adam(lr=args.lr),metrics=['accuracy'])
if args.crossvalid:
	skf = StratifiedKFold(labels, n_folds=10, shuffle=True)

	
else:
	# Train
	hist = model.fit(x_train,y_train,epochs=args.epochs,batch_size=128,shuffle='batch')

	# Test
	y_predict = model.predict(x_test)
	test_pred = pd.DataFrame({'predict':np.squeeze(y_predict),'test':y_test})

	# Save model
	model.save(args.outfile + '.mdl')

	# Save model summary
	with open(args.outfile + '_report.txt','w') as fh:
	    # Pass the file handle in as a lambda function to make it callable
	    model.summary(print_fn=lambda x: fh.write(x + '\n'))

	# Save model history
	with open(args.outfile + '_hist.json', 'w') as f:
	    json.dump(hist.history, f)

	# Save predictions on test set
	test_pred.to_csv(args.outfile + '_test_pred.csv')

          
          



