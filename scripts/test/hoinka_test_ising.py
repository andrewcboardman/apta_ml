import pandas as pd
from storage.code import encode_simple
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

base_dict = {'A':0,'C':1,'G':2,'T':3}
hoinka_data = pd.read_csv('/storage/data/Hoinka_samples.csv')
seqs = [list(x) for x in hoinka_data['Consensus sequence']]
features = to_categorical(encode_simple.vec_translate(seqs,base_dict))
labels = hoinka_data['Kd'].values
train_data,test_data,train_labels,test_labels = train_test_split(hoinka_data,labels)

from storage.code import mf_ising
couplings= pd.read_csv('/storage/mf_ising/Ilk_pool_J.txt',sep=' ',header=None).values
fields = pd.read_csv('/storage/mf_ising/Ilk_pool_h.txt',sep=' ',header=None).values

pred = mf_ising.ising_eval(features,fields,couplings.reshape(40,4,40,4))
fit = np.polyfit(pred,labels,1)
fit_Kd = fit[0]*pred+fit[1]
np.savetxt('/storage/mf_ising/hoinka_test.txt',np.stack((pred,labels),axis=1))