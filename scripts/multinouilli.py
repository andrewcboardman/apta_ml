import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_actives(csv_file):
    raw_data = pd.read_csv(csv_file,nrows=1000000,header=None)
    data = raw_data[raw_data[40]==1].drop(40,axis=1)
    for i in range(1,20):
        try:
            raw_data = pd.read_csv(csv_file,nrows=1000000,skiprows=i*1000000,header=None)
            data = pd.concat((data,raw_data[raw_data[40]==1].drop(40,axis=1)),axis=0)
        except pd.errors.EmptyDataError:
            return data
            
def one_hot(data):
    data_one_hot = np.zeros((len(data),40,4))
    idx_0 = np.arange(len(data))[:,np.newaxis]
    idx_1 = np.arange(40)[np.newaxis,:]
    data_one_hot[idx_0,idx_1,data] = 1
    return data_one_hot

def ML_fit(data):
    frequencies = np.mean(data,axis=0)
    fields = -np.log(frequencies / frequencies[0,:]).T
    return fields

def ML_eval(data,fields):
    return np.tensordot(data,fields,axes=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--infile',action='store',help='Input file')
    parser.add_argument('-o','--outfile',action ='store',help='Output file')
    args = parser.parse_args()
    
    data = load_actives(args.infile)
    data_one_hot = one_hot(data)
    
    train,test = train_test_split(data_one_hot)
    fields = ML_fit(train)
    
    random = one_hot(np.random.randint(4,size=(len(test),40,4)))
    
    E_test = ML_eval(test,fields)
    E_random = ML_eval(random,fields)
    
    np.savetxt(args.outfile,np.stack((E_test,E_random),axis=0))
    
if __name__ == '__main__':
    main()