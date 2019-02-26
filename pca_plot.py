from sklearn.decomposition import PCA
import argparse
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--infile', type=str, action='store', dest='infile',	help='FASTA-formatted samples')
parser.add_argument('-o', '--outfile', type=str, action='store', dest='outfile', help='name of encoded text file')
args = parser.parse_args()
data = np.genfromtxt(args.infile)

pca = PCA(n_components=10)
pc_data = pca.fit_transform(data)

plt.scatter(pc_data[:,0],pc_data[:,1])
plt.xlabel('PC 1')
plt.ylabel('PC_2')
plt.show()
