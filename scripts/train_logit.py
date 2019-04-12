import argparse
import numpy as np
from keras.models import Sequential,load_model
from keras.layers import Conv1D, MaxPooling1D, Dense, Dropout,Flatten,GlobalAveragePooling1D
from keras.optimizers import Adam
import json
from sklearn.model_selection import KFold,train_test_split
import pickle
import h5py
from keras.utils.io_utils import HDF5Matrix
import pandas as pd
from keras.utils import Sequence
from copy import copy

class HDF5Sequence(Sequence):
	def __init__(self, x_set, y_set, batch_size):
		self.x, self.y = x_set, y_set
		self.batch_size = batch_size
		self.L = int(np.ceil(len(self.x) / float(self.batch_size)))
		self.data_idxs = [(idx * self.batch_size,(idx + 1) * self.batch_size) for i in range(self.L)] + [(idx * self.L,len(self.x))]

	def __len__(self):
		return self.L

	def __getitem__(self,  batch_idx):
		batch_x = self.x[self.data_idxs[batch_idx][0]:self.data_idxs[batch_idx][1]]
		batch_y = self.y[self.data_idxs[batch_idx][0]:self.data_idxs[batch_idx][1]]
		return (batch_x,batch_y)

	def drop(indices):
		for k in indices:
			self.L -= 1
			self.ix.pop(k)

	def CrossValSplit(seq,n_splits):
		kf = KFold(n_splits=nfolds)
		batch_ixs = np.arange(seq.L)
		for train_batches,test_batches in kf.split(batch_ixs):
			train_seq = HDF5Sequence(seq.x,seq.y,seq.batch_size)
			test_seq = HDF5Sequence(seq.x,seq.y,seq.batch_size)
			train_seq.drop(test_batches)
			test_seq.drop(train_batches)
			yield (train_seq,test_seq)

def create_model(loadmodel,conv_n,conv_w,dr):
    if loadmodel == None:
        # Build network
        model = Sequential()
        model.add(Conv1D(conv_n, conv_w, activation='relu', input_shape=(40, 4)))
        model.add(Conv1D(conv_n, conv_w, activation='relu'))
        model.add(MaxPooling1D(3))
        model.add(Dropout(dr))
        model.add(Conv1D(2*conv_n, conv_w, activation='relu'))
        model.add(Conv1D(2*conv_n, conv_w, activation='relu'))
        model.add(GlobalAveragePooling1D())
        model.add(Dropout(dr))
        model.add(Dense(1,activation='sigmoid'))
        return model
    else:
        model = load_model(loadmodel)
        return model

def train_model(model,train_seq,lr,epochs):
    model.compile(loss='binary_crossentropy',optimizer=Adam(lr=lr),metrics=['accuracy'])
    return model.fit_generator(train_seq,epochs=epochs,shuffle=True)

def evaluate_model(model,test_seq,outfile):
    # Test
    for i, batch in enumerate(test_seq):
        if i//1000 == 0:
            print('Batch', i, 'of', len(test_seq))
        predict = model.predict(batch[0])
        test_pred = pd.DataFrame({'predict':np.squeeze(predict),'test':batch[1]})
        # Save predictions on test set
        test_pred.to_csv(outfile + '_test_pred.csv',mode='a',header=False)

def save_model_stats(model,hist,outfile):
    # Save model
    model.save(outfile + '.mdl')

    # Save model summary
    with open(outfile + '_report.txt','w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

    # Save model history
    with open(outfile + '_hist.json', 'w') as f:
        json.dump(hist.history, f)

def main():
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
    parser.add_argument('-l','--loadmodel',action='store',help='File to load model from')
    args = parser.parse_args()

    # Import data
    x_train = HDF5Matrix(args.infile,'/x_train')
    x_test = HDF5Matrix(args.infile,'/x_test')
    y_train = HDF5Matrix(args.infile,'/y_train')
    y_test = HDF5Matrix(args.infile,'/y_test')

    if args.crossvalid:
    	train_seq, test_seq = 
        for i in range
            print("Running Fold", i+1, "/", 10)

            model = create_model(args.loadmodel,args.conv_n,args.conv_w,args.dr)
            hist = train_model(model,x_train[train_index],y_train[train_index],args.lr,args.epochs)
            evaluate_model(model, hist, x_train[test_index], y_train[test_index],args.outfile+'_{}'.format(i+1))
            del model
            del hist
    else:
        train_seq = HDF5Sequence(x_train,y_train,128)
        test_seq = HDF5Sequence(x_test,y_test,128)
        model = create_model(args.loadmodel,args.conv_n,args.conv_w,args.dr)
        hist = train_model(model,train_seq,args.lr,args.epochs)
        evaluate_model(model, test_seq, args.outfile)
        save_model_stats(model, hist,outfile)
        
if __name__ == '__main__':
    main()
