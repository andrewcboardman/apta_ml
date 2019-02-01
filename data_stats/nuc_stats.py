import matplotlib.pyplot as plt

data = Data(['mf_artificial_samples.fasta'],'ACGT')
profile = sum((data.astype('int32') for data in data.data))/len(data.data)
plt.switch_backend('TkAgg')
plt.bar(range(1,41),profile[:,0])
plt.bar(range(1,41),profile[:,1],bottom=profile[:,0])
plt.bar(range(1,41),profile[:,2],bottom=profile[:,0]+profile[:,1])
plt.bar(range(1,41),profile[:,3],bottom=profile[:,0]+profile[:,1]+profile[:,2])
plt.legend(('A','C','G','T')
	)
plt.title('Nucleotide distribution for the artificially generated sequences')
plt.show()