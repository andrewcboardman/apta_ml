import h5py
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--infile',action='store',help='Input file')
    parser.add_argument('-o','--outfile',action ='store',help='Output file')
    args = parser.parse_args()
    
    with h5py.File(args.infile) as fh:
        X = fh['/x_train'][:1000].reshape(1000,160)
        y = fh['/y_train'][:1000]
        
    dcp = PCA()
    X_t = dcp.fit_transform(X)
    plt.scatter(X_t[:,0],X_t[:,1],c=y)
    plt.savefig(args.outfile)
    
if __name__ == '__main__':
    main()