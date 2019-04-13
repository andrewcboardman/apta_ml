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

def inv_covar(data_one_hot):
    data_shape = data_one_hot[:,:,1:].reshape(len(data_one_hot),120)
    covar = data_shape.T @ data_shape / len(data_one_hot)
    couplings = np.zeros((40,4,40,4))
    couplings[:,1:,:,1:] = np.linalg.inv(covar).reshape(40,3,40,3)
    return couplings

def ising_fields(data,couplings):
    frequencies = np.mean(data,axis=0)
    fields = np.log(frequencies / frequencies[:,0][:,np.newaxis]) #- np.tensordot(couplings,frequencies,axes=2)
    return fields

def ising_eval(data,fields,couplings):
    return - np.tensordot(data,fields,axes=2) - np.einsum('ijk,jklm,ilm->i',data,couplings,data)/2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--infile',action='store',help='Actives')
    parser.add_argument('-o','--outfile',action ='store',help='Output file')
    parser.add_argument('--inactives',action='store',help='Inactives')
    args = parser.parse_args()
    
    data = pd.read_csv(args.infile,header=None).drop(40,axis=1)
    data_one_hot = one_hot(data)
        
    train,test = train_test_split(data_one_hot)
    couplings = -inv_covar(train)
    fields = ising_fields(train,couplings)
    random = one_hot(np.random.randint(4,size=(len(test),40)))
    
    E_test = ising_eval(test,fields,couplings)
    E_random = ising_eval(random,fields,couplings)
    
    data = pd.read_csv(args.inactives,header=None,nrows=len(test)).drop(40,axis=1)
    data_one_hot = one_hot(data)
    E_inactives = ising_eval(data_one_hot,fields,couplings)
    
    np.savetxt(args.outfile+'_h.txt',fields)
    np.savetxt(args.outfile+'_J.txt',couplings.reshape(160,160))    
    np.savetxt(args.outfile+'_E.txt',np.stack((E_test,E_random,E_inactives),axis=1))
    
if __name__ == '__main__':
    main()

