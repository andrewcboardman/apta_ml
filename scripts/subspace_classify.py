import argparse
import numpy as np
import matplotlib.pyplot as plt

def dist_subspace(data, means, stds, vects):
    clean_data = data[:,stds!=0]
    scaled_data = (clean_data - means[stds!=0]) / stds
    proj = (vects @ (scaled_data @ vects).T).T
    return np.sqrt(np.sum((scaled_data - proj)**2,axis=1))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i1','--infile1',action='store',help='Test file (actives)')
    parser.add_argument('-i2','--infile2',action='store',help='Test file (inactives)')
    parser.add_argument('-v1','--vects1',action='store',help='Active means, stds, and eigenvectors')
    parser.add_argument('-v2','--vects2',action='store',help='Inactive means, stds, and eigenvectors')
    args = parser.parse_args()

    
    data_a = np.genfromtxt(args.infile1).astype('int8')
    data_i = np.genfromtxt(args.infile2).astype('int8')
    with np.load(args.vects1) as fh_a:
        means_a = fh_a['means']
        stds_a = fh_a['stds']
        vects_a = fh_a['evects'][:,fh_a['evals']>fh_a['bound']]
    with np.load(args.vects2) as fh_i:
        means_i = fh_i['means']
        stds_i = fh_i['stds']
        vects_i = fh_i['vects'][:,fh_i['evals']>fh_i['bound']]

    # Distance of active test data from active subspace
    dist_a_a = dist_subspace(data_a,means_a,stds_a,vects_a)
    # Distance of active test data from inactive subspace
    dist_a_i = dist_subspace(data_a,means_i,stds_i,vects_i)
    # Distance of inactive test data from active subspace
    dist_i_a = dist_subspace(data_i,means_a,stds_a,vects_a)
    # Distance of inactive test data from inactive subspace
    dist_i_i = dist_subspace(data_i,means_i,stds_i,vects_i)

    thresh = 100
    thresholds = np.arange(-thresh,thresh,0.01)
    true_pos_rate = np.zeros_like(thresholds)
    false_pos_rate = np.zeros_like(thresholds)

    for j,t in enumerate(thresholds):
    	true_pos_rate[j] = np.sum(dist_a_a < dist_a_i + t)/len(data_a)
    	false_pos_rate[j] = np.sum(dist_i_a < dist_i_i + t)/len(data_i)

    plt.plot(true_pos_rate, false_pos_rate)
    plt.show()

if __name__ =='__main__':
    main()
