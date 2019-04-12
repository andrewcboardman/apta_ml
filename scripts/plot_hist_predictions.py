import numpy as np 
import matplotlib.pyplot as plt
predictions = np.squeeze(np.load('/home/andrew/Downloads/Ilk_Pool_deep_predictions.npy'))
labels = np.squeeze(np.load('/home/andrew/Downloads/Ilk_Pool_deep_test.npy'))

actives = predictions[labels == 0]
plt.hist(np.log(actives),bins=100,range=(-1,0))
plt.show()