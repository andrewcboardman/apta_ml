import matplotlib.pyplot as plt
from scipy.signal import correlate
import numpy as np
def PlotAutoCorr(samples,n,L):
	# normalise samples
	normed_samples = (samples - np.mean(samples,axis=0))/np.std(samples,axis=0)
	# calculate autocorrelation
	corr = correlate(normed_samples,normed_samples,mode='same',method='fft')/(n*L*3)
	print(corr.shape)
	return plt.plot(range(-n//2,n//2),corr)


energy = np.genfromtxt('data/fake/sk_energies.txt')
#seqs = np.genfromtxt('data/fake/sk_encode.txt',dtype=int)

PlotAutoCorr(energy,100000,1)
#PlotAutoCorr(seqs[:,0],100000,1)
plt.savefig('data/fake/sk_autocorr.png')