import numpy as np
import matplotlib.pyplot as plt

J = np.genfromtxt('/home/andrew/Downloads/Ilk_pool_J.txt')
M = np.load('/home/andrew/Downloads/mutual_inf_Ilk_pool.npy')
J = J.reshape(40,4,40,4)
J[range(40),:,range(40),:] = 0
J = np.sum(J**2,axis=(1,3))
# scale = np.std(J.flatten())/np.std(M.flatten())
# for i in range(40):
# 	for j in range(i+1,40):
# 		J[i,j] = M[i,j]*scale
plt.imshow(J,vmax=4,cmap='inferno')
plt.show()
#savefig('Ilk_pool_couplings.png')

# h = np.genfromtxt('/home/andrew/Downloads/Ilk_pool_h.txt')
# plt.imshow(h.T,cmap='inferno')
# plt.show()