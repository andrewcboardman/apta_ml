import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
x_train = np.load('data/IgE_train_features_stand.npy')
y_train = np.load('data/IgE_train_labels.npy')
x_test = np.load('data/IgE_test_features_stand.npy')
y_test = np.load('data/IgE_test_labels.npy')




model = PCA(n_components=10)
x_train_pca = model.fit_transform(x_train)
x_train_pca_bind = x_train_pca[y_train.astype(bool)]
x_train_pca_control = x_train_pca[np.logical_not(y_train.astype(bool))]

fig, axs = plt.subplots(1,2)
h1,_,_ = plt.hist2d(x_train_pca_bind[:,0],x_train_pca_bind[:,1],label='Binding sequences')
axs[0].imshow(h1)
h2,_,_ = plt.hist2d(x_train_pca_control[:,0],x_train_pca_control[:,1],label='Control sequences')

plt.title('Principal components plot of binding and control sequences')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.savefig('pca_plot.png')

#plt.bar(range(10),model.explained_variance_)
plt.show()