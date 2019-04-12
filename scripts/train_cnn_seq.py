import argparse
import numpy as np
from keras.models import Sequential,load_model
from keras.layers import Conv1D, MaxPooling1D, Dense, Dropout,Flatten,GlobalAveragePooling1D, Reshape
from keras.regularizers import l1_l2
from keras.optimizers import Adam

import json
from sklearn.model_selection import KFold,train_test_split
import pickle
import h5py
from keras.utils.io_utils import HDF5Matrix
import pandas as pd
from keras.utils import Sequence
from copy import copy
from keras.callbacks import EarlyStopping

class HDF5Sequence(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.L = int(np.ceil(len(self.x) / float(self.batch_size)))
        self.data_idxs = [(idx * self.batch_size,(idx + 1) * self.batch_size) for idx in range(self.L - 1)] + [(self.batch_size * (self.L - 1),len(self.x))]

    def __len__(self):
        return self.L

    def __getitem__(self, batch_idx):
        batch_x = self.x[self.data_idxs[batch_idx][0]:self.data_idxs[batch_idx][1]]
        batch_y = self.y[self.data_idxs[batch_idx][0]:self.data_idxs[batch_idx][1]]
        return (batch_x,batch_y)

    def drop(self, indices):
        for k in indices:
            self.L -= 1
            self.data_idxs.pop(k)

def CrossValSplit(seq,nfolds):
    kf = KFold(n_splits=nfolds)
    batch_ixs = np.arange(seq.L)
    splits = [x for x in kf.split(batch_ixs)]
    train_seqs = [HDF5Sequence(seq.x,seq.y,seq.batch_size) for i in range(nfolds)]
    test_seqs = [HDF5Sequence(seq.x,seq.y,seq.batch_size) for i in range(nfolds)]
    for i in range(nfolds):
        train_batches = np.flip(splits[i][0],axis=0)
        test_batches = np.flip(splits[i][1],axis=0)
        train_seqs[i].drop(test_batches)
        test_seqs[i].drop(train_batches)
    return (train_seqs,test_seqs)

def TrainValSplit(seq,nfolds):
    kf = KFold(n_splits=nfolds)
    batch_ixs = np.arange(seq.L)
    splits = next(kf.split(batch_ixs))
    train_seq = HDF5Sequence(seq.x,seq.y,seq.batch_size)
    test_seq = HDF5Sequence(seq.x,seq.y,seq.batch_size)
    train_batches = np.flip(splits[0],axis=0)
    test_batches = np.flip(splits[1],axis=0)
    train_seq.drop(test_batches)
    test_seq.drop(train_batches)
    return (train_seq,test_seq)

def create_model(model_name,conv_n,conv_w,dr):
    if model_name == 'logit':
        model = Sequential()
        model.add(Reshape((160,),input_shape=(40,4)))
        model.add(Dense(1, activation='sigmoid', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))
    elif model_name == 'shallow':
        model = Sequential()
        model.add(Conv1D(conv_n, conv_w, activation='relu', input_shape=(40, 4)))
        model.add(Conv1D(conv_n, conv_w, activation='relu'))
        model.add(GlobalAveragePooling1D())
        model.add(Dropout(dr))
        model.add(Dense(1,activation='sigmoid'))
    elif model_name == 'deep':
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
    elif model_name == 'deep_dense':
        model = Sequential()
        model.add(Conv1D(conv_n, conv_w, activation='relu', input_shape=(40, 4)))
        model.add(Dense(100))
        model.add(Dropout((dr)))
        model.add(Dense(100))
        model.add(Dense(100))
        model.add(Dropout(dr))
        model.add(Dense(1,activation='sigmoid'))
    elif model_name == 'recurrent':
        model.add(Embedding(max_features, 128, input_shape=(40,4)))
        model.add(Bidirectional(LSTM(64)))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
    else:
        model = load_model(loadmodel)
    return model

def train_model(model, train_seq, val_seq, lr, epochs):
    model.compile(loss='binary_crossentropy',optimizer=Adam(lr=lr),metrics=['accuracy'])
    return model.fit_generator(train_seq,epochs=epochs,shuffle=True,validation_data=val_seq,validation_steps=1,callbacks=[EarlyStopping()])

def evaluate_model(model, test_seq, outfile):
    """Calculate accuracy"""
    return model.evaluate_generator(test_seq)

def predict_model(model, test_seq, outfile):
    predictions = (model.predict_generator(test_seq,steps=1) for i in range(len(test_seq)))


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
    parser.add_argument('-m','--model_type',action='store',default='logit',help='File to load model from')
    parser.add_argument('-t','--test',action='store_true',default=False,help='Record predictions')
    args = parser.parse_args()

    # Import data
    x_train = HDF5Matrix(args.infile,'/x_train')
    x_test = HDF5Matrix(args.infile,'/x_test')
    y_train = HDF5Matrix(args.infile,'/y_train')
    y_test = HDF5Matrix(args.infile,'/y_test')

    if args.crossvalid:
        train_seq = HDF5Sequence(x_train,y_train,128)
        nfolds = 10
        train_cvs,test_cvs = CrossValSplit(train_seq,nfolds)
        for i in range(nfolds):
            print("Running Fold", i+1, "/", nfolds)
            model = create_model(args.model_type,args.conv_n,args.conv_w,args.dr)
            hist = train_model(model,train_cvs[i],test_cvs[i],args.lr,args.epochs)
            test_acc = evaluate_model(model, test_cvs[i], args.outfile+'_{}'.format(i+1))
            save_model_stats(model, hist, args.outfile+'_{}'.format(i+1))
    if args.testmodel:
        model = create_model(args.model_type,args.conv_n,args.conv_w,args.dr)

    else:
        train_seq = HDF5Sequence(x_train,y_train,128)
        train_split,val_split = TrainValSplit(train_seq,10)
        test_seq = HDF5Sequence(x_test,y_test,128)
        model = create_model(args.model_type,args.conv_n,args.conv_w,args.dr)
        hist = train_model(model,train_split,val_split,args.lr,args.epochs)
        test_acc = evaluate_model(model, val_split, args.outfile)
        save_model_stats(model, hist, args.outfile)

if __name__ == '__main__':
    main()