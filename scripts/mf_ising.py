import argparse
import pandas as pd
import numpy as np

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

def inv_covar(data):
    data_shape = data.reshape(len(data),160)
    covar = data_shape.T @ data_shape / len(data)
    return np.linalg.inv(covar).reshape(40,4,40,4)

def ising_fields(data,couplings):
    frequencies = np.mean(data,axis=0)
    fields = -np.log(frequencies / frequencies[0,:]).T - 
    return fields

def ising_eval(data,fields,couplings):
    # Needs testing
    return np.tensordot(data,fields,axes=2) + np.einsum('ijk,jklm,ilm',data,couplings,data)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--infile',action='store',help='Input file')
    parser.add_argument('-o','--outfile',action ='store',help='Output file')
    parser.add_argument('--inactives',action='store',help='Inactives')
    args = parser.parse_args()
    
    data = load_actives(args.infile)
    data_one_hot = one_hot(data)
        
    train,test = train_test_split(data_one_hot)
    couplings = ising_mf(train)
    fields = ising_fields(train,couplings)
    
    E_test = ising_eval(test,fields,couplings)
    E_random = ising_eval(random,fields,couplings)
    
    data = pd.read_csv(args.inactives,header=None,nrow=1000000)
    E_inactives = ising_eval(data,fields,couplings)
    
    np.savetxt(args.outfile,inv_covar)
    
if __name__ == '__main__':
    main()

