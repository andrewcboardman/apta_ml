import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


raw_data = pd.read_csv('~/Downloads/Ilk_pool_mf_E.txt',header = None,sep = ' ')
raw_data.columns = ['Pool 5','Random','Pool 2']
bin_edges = np.linspace(-np.max(raw_data.values),-np.min(raw_data.values),200)

fig, axs = plt.subplots(3,1)
h0 = axs[0].hist(-raw_data['Pool 5'], label = 'Active', bins = bin_edges, density = True, color='red')
h1 = axs[1].hist(-raw_data['Pool 2'], label = 'Decoy', bins = bin_edges, density = True, color='green')
h2 = axs[2].hist(-raw_data['Random'], label = 'Random', bins = bin_edges, density = True, color='blue')
for ax in axs:
	ax.set_xlim([-100,0])
	ax.set_ylim([0,0.08])
	ax.legend()
	ax.set_ylabel('Density')
axs[0].tick_params(axis = 'x', bottom = False, labelbottom = False)
axs[1].tick_params(axis = 'x', bottom = False, labelbottom = False)
axs[2].set_xlabel('Scaled log probability')
#plt.show()
plt.savefig('figs/Ilk_pool_mf_hist.png')