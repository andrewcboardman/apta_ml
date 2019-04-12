import numpy as np
from scipy.stats import zscore

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--infile',action='store',help='Input file')
    parser.add_argument('-o','--outfile',action='store',help='Output file')
    parser.add_argument('-k',action='store',help='Length of words to be extracted')
    args = parser.parse_args()

    # Load data
    data = np.genfromtxt(args.infile).astype('int8')

    # Remove constant columns
    sum_data = np.sum(data,axis=1)
    zero_cols = sum_data == 0
    one_cols = sum_data == len(data)
    clean_data = data[:,np.logical_or(np.logical_not(zero_cols),np.logical_not(one_cols))]

    # Transform to z-scores
    scale_data = zscore(data, axis=1, ddof=1)

    # Covariance matrix
    covar = scale_data @ scale_data.T / N

    # Eigendecomposition of covariance matrix
    evals, evects = np.linalg.eig(covar)

    # Calculate Marchenko-Pastur bound
    p = data.shape[1]
    n = data.shape[0]
    bound = (1 + np.sqrt(p/n))**2
    num_eig = np.sum(evals > bound)

    print("{} eigenvalues above the Marchenko-Pastur bound".format(num_eig))

    # Save significant eigenvectors
    np.savetxt(args.outfile,evects[:,evals > bound]


if __name__ =='__main__':
    main()