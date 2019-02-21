import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

path = 'data/IgE'
x_train = np.load(f'{path}/train_features_stand.npy')
y_train = np.load(f'{path}/train_labels.npy')
x_test = np.load(f'{path}/test_features_stand.npy')
y_test = np.load(f'{path}/test_labels.npy')

model = PCA(n_components=10)
x_train_pca = model.fit_transform(x_train)
x_train_pca_bind = x_train_pca[y_train.astype(bool)]
x_train_pca_control = x_train_pca[np.logical_not(y_train.astype(bool))]

fig, axs = plt.subplots()
plt.scatter(x_train_pca_bind[:,0],x_train_pca_bind[:,1],label='Binding sequences',alpha=0.5)

plt.scatter(x_train_pca_control[:,0],x_train_pca_control[:,1],label='Control sequences',alpha=0.5)
# axs[1].imshow(h2)
plt.title('Principal components plot of binding and control sequences')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend()
plt.savefig(f'{path}/pca_plot.png')
