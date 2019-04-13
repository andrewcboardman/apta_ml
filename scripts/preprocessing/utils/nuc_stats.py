import matplotlib.pyplot as plt
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--infile', type=str, action='store', dest='infile',	help='FASTA-formatted samples')
parser.add_argument('-o', '--outfile', type=str, action='store', dest='outfile', help='name of encoded text file')
args = parser.parse_args()


data = np.genfromtxt(args.infile)
data = data.reshape(data.shape[0],40,3)

profile = np.mean(data,axis=0)

plt.bar(range(1,41),profile[:,0])
plt.bar(range(1,41),profile[:,1],bottom=profile[:,0])
plt.bar(range(1,41),profile[:,2],bottom=profile[:,0]+profile[:,1])
plt.bar(range(1,41),profile[:,3],bottom=profile[:,0]+profile[:,1]+profile[:,2])
plt.legend(('A','C','G','T')
plt.title('Nucleotide distribution')
plt.savefig(args.outfile)