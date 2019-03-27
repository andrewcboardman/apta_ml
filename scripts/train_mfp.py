import numpy as np
from mf_potts import MFPottsClassifier
import joblib
import h5py
import argparse

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
		cE = -100*np.einsum('ijk,jklm,ilm->i',samples,self.J,samples)/2
		return fE + cE

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
		
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-i','--infile',action='store',help='input file')
	args = parser.parse_args()

	fh = h5py.File(args.infile,'r')
	x_train = fh['/x_train']
	y_train = fh['/y_train']

	model = MFPottsClassifier()
	model.fit(x_train,y_train)
	model.save('mfp_2.joblib')

	# model = joblib.load('mfp.joblib')
	y_pred = model.predict(x_test)
	np.savetxt('mfp_2_logit_test_pred.txt',np.stack((y_test,y_pred)).T)

	from sklearn.metrics import roc_curve,roc_auc_score
	auc = roc_auc_score(y_test,y_pred)
	fpr,tpr,_= roc_curve(y_test,y_pred)
	from matplotlib import pyplot as plt
	plt.plot(fpr,tpr)
	plt.fill_between(fpr,tpr,alpha=0.5)
	plt.xlabel('FPR')
	plt.ylabel('TPR')
	plt.title(f'ROC for classification using mean-field potts\n AUC = {auc}')
	plt.savefig('test_roc_mfp_2')