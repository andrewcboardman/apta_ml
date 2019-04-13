from keras.models import load_model
from keras.utils import to_categorical
import numpy as np
import argparse
import matplotlib.pyplot as plt
import pandas as pd

def vec_translate(a,mydict):
    return np.vectorize(mydict.__getitem__)(a)

def init(N):
    return to_categorical(np.random.randint(4,size=(N,40)),num_classes=4)

def GibbsSample(old_samples,model,L,N,beta):
    # Propose changes to sequence
    new_samples = np.copy(old_samples)
    new_samples[range(N),np.random.randint(L,size=N),:] = to_categorical(np.random.randint(4,size=(N,)),num_classes=4)

    # Calculate changes in log probability
    E_old = -np.log(model.predict(old_samples))
    E_new = -np.log(model.predict(new_samples))

    # Accept or reject changes based on energy change
    dE = E_new - E_old
    accept = dE < 0
    accept[dE > 0] = np.exp(-dE[dE > 0] * beta) > np.random.rand(np.sum(dE > 0))
    
    
    return (E_new,np.where(accept[...,np.newaxis],new_samples,old_samples))

def samples(model,init_samples,n_warmup_steps,n_samples,n_steps,beta):
    all_samples = np.zeros((n_samples,*init_samples.shape))
    all_energies = np.zeros((n_samples,init_samples.shape[0]))
    N = init_samples.shape[0]
    L = init_samples.shape[1]
    samples = np.copy(init_samples)
    for i in range(n_warmup_steps):
        _,samples = GibbsSample(samples,model,L,N,beta)
    for i in range(n_samples):
        for j in range(n_steps):
            energies, samples = GibbsSample(samples,model,L,N,beta)
        all_energies[i] = np.squeeze(energies)
        all_samples[i] = samples
    return (all_energies,all_samples)
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--model_file',action='store',help='Model file')
    parser.add_argument('--n_warmup_steps',action='store',type=int,help='Number of warmup steps')
    parser.add_argument('--n_samples',action='store',type=int,help='Number of samples')
    parser.add_argument('--n_steps',action='store',type=int,help='Number of steps between samples')
    parser.add_argument('-b','--beta',action='store',type=float,help='Inverse temperature')
    parser.add_argument('-o','--outfile',action='store',help='Output file')
    args = parser.parse_args()
    
    model = load_model(args.model_file)
    base_dict = {'A':0,'C':1,'G':2,'T':3}
    hoinka_data = pd.read_csv('/storage/data/Hoinka_samples.csv')
    seqs = [list(x) for x in hoinka_data['Consensus sequence']]
    init = to_categorical(vec_translate(seqs,base_dict))
    
    all_energies, all_samples = samples(model,init,0,args.n_samples,args.n_steps,args.beta)
    np.save(args.outfile,all_samples)
    np.save(args.outfile + '_E', all_energies)
    
if __name__ == '__main__':
    main()
        
    
    
    