import numpy as np

class MFPotts():
	def fit(self,data):
		# Select training data
		n_train,n_spins,n_states = data.shape
		# One-point frequencies
		f1s = np.mean(data,axis=0)
		# Two-point frequencies
		f2s = sum((np.outer(x,x) for x in data))/n_train
		# Covariance
		cov = f2s - np.outer(f1s,f1s)
		# Mean-field couplings
		self.J = -np.linalg.inv(cov).reshape(n_spins,n_states,n_spins,n_states)
		# Set self-couplings to zero
		for i in range(n_spins):
			self.J[i,:,i,:] = 0
		# mean-field fields
		self.h = np.log(f1s / np.clip((1 - np.sum(f1s,axis=1)),0.001,None).reshape(n_spins,1)) \
		- np.tensordot(self.J,f1s,axes=2)
	def get_fields(self):
		return self.h
	def get_couplings(self):
		return self.J
	def energies(self,samples):
		# Field-related energies
		fE = -np.tensordot(samples,self.h,axes=2)
		# Coupling-related energies
		cE = -np.einsum('ijk,jklm,ilm->i',samples,self.J,samples)/2
		return fE + cE
	def save(self,filename):
		joblib.dump(self,filename)

class MFPottsClassifier():
	def fit(self,features,labels):
		self.model_plus = MFPotts()
		self.model_minus = MFPotts()
		self.model_plus.fit(features[labels.astype(bool)])
		self.model_minus.fit(features[np.logical_not(labels.astype(bool))])
		energies = self.score(features)
		self.clf = LogisticRegression()
		self.clf.fit(energies,labels)
	def predict(self,features):
		energies = self.score(features)
		return self.clf.predict(energies)
	def score(self,features):
		return (self.model_plus.energies(features) - self.model_minus.energies(features))[:,np.newaxis]
	def save(self,filename):
		joblib.dump(self,filename)
		
class CNNClassifier(Sequential):
	def add_layers(conv_n,conv_w,dr):
		self.add(Conv1D(conv_n, conv_w, activation='relu', input_shape=(40, 4)))
		self.add(Conv1D(conv_n, conv_w, activation='relu'))
		self.add(MaxPooling1D(3))
		self.add(Dropout(args.dr))
		self.add(Conv1D(2*conv_n, conv_w, activation='relu'))
		self.add(Conv1D(2*conv_n, conv_w, activation='relu'))
		self.add(GlobalAveragePooling1D())
		self.add(Dropout(dr))
		self.add(Dense(1,activation='sigmoid'))


