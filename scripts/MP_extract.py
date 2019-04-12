import numpy as np
from scipy.stats import zscore
import argparse
import time

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--infile',action='store',help='Input file')
    parser.add_argument('-o','--outfile',action='store',help='Output file')
    args = parser.parse_args()
    t0 = time.time()
    # Load data
    data = np.genfromtxt(args.infile).astype('int8')

    # Calculate & save mean and variance of data
    means = np.mean(data,axis=0)
    stds = np.std(data,axis=0)

    t1 = time.time()
    print('Calculated means and standard deviation in {}s'.format(t1-t0))
    # Transform to z-scores
    clean_data = data[:,stds!=0]
    scale_data = (clean_data - means[stds!=0]) / stds

    # Covariance matrix
    covar = scale_data.T @ scale_data / len(data)
    t2 = time.time()
    print('Got covariance matrix in {}s'.format(t2-t1))
    print(covar.shape)

    # Eigendecomposition of covariance matrix
    evals, evects = np.linalg.eig(covar)
    t3 = time.time()
    print('Diagonalised covariance matrix in {}s'.format(t3-t2))
    # Calculate Marchenko-Pastur bound
    p = data.shape[1]
    n = data.shape[0]
    bound = (1 + np.sqrt(p/n))**2
    num_eig = np.sum(evals > bound)

    print("{} eigenvalues above the Marchenko-Pastur bound".format(num_eig))

    # Save means, stds, eigenvalues, eigenvectors
    np.savez(args.outfile,means=means,stds=means,bound=bound,evals=evals,evects=evects)

if __name__ == '__main__':
    main()