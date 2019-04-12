import os

from Bio import SeqIO
import argparse
def lab(rec, label):
	rec.id = rec.id + str(label)
	return rec

def label_fasta(infile, outfile, label):
	# load data from FASTA files
	seqs = SeqIO.parse(infile,'fasta')
	# label 
	seqs_l = (lab(rec,label) for rec in seqs)
	# write
	SeqIO.write(seqs_l,outfile,'fasta')

def merge_fasta(infile1, infile2, outfile):
	os.system('cat {} {} | perl seq-shuf.pl > {}'.format(infile1,infile2,outfile))

def encode(rec,mydict):
	seq = str(rec.seq)
	ix = str(rec.id)
	chars = np.array(list(seq))
	code = np.vectorize(mydict.__getitem__)(chars)
	label = int(ix[-1])
	return (to_categorical(code,num_classes=4,dtype='int8'),label)
	
def to_hdf5(infile,outfile):

	base_dict = {'A':0,'C':1,'G':2,'T':3}

	# load data from FASTA files
	seqs = SeqIO.parse(infile,'fasta')

	# count length of sequences
	first_seq = next(seqs)
	L = len(first_seq.seq)
	
	# count lines in file
	N = sum(1 for seq in seqs) + 1

	print('Counted sequences. Writing to hdf5...')

	# Reload data
	seqs = SeqIO.parse(infile,'fasta')

	# Translate to 
	seqs_encode = (encode(rec,base_dict) for rec in seqs)
	# Train-test split
	n_train = int(np.round(0.8*N))
	n_test = N - n_train

	
	with h5py.File(outfile,'w') as outf:
		# Allocate memory
		outf.create_dataset('/x_train',shape=(n_train,L,4),dtype='i1')
		outf.create_dataset('/x_test',shape=(n_test,L,4),dtype='i1')
		outf.create_dataset('/y_train',shape=(n_train,),dtype='i1')
		outf.create_dataset('/y_test',shape=(n_test,),dtype='i1')
		# Go through file and write
		for i,(x,y) in enumerate(seqs_encode):
			if i < n_train:
				outf['/x_train'][i,...] = x
				outf['/y_train'][i] = y
			else:
				outf['/x_test'][i-n_train,...] = x
				outf['/y_test'][i-n_train] = y
def to_txt(infile,outfile):

	base_dict = {'A':0,'C':1,'G':2,'T':3}

	print('Counted sequences. Writing to hdf5...')

	# Reload data
	seqs = SeqIO.parse(infile,'fasta')

	# Translate to 
	seqs_encode = (encode(rec,base_dict) for rec in seqs)
	
	# Train-test split
	n_train = int(np.round(0.8*N))
	n_test = N - n_train
	
	with open(outfile,'w') as outf:
		# Allocate memory
		outf.write()


		outf.create_dataset('/x_train',shape=(n_train,L,4),dtype='i1')
		outf.create_dataset('/x_test',shape=(n_test,L,4),dtype='i1')
		outf.create_dataset('/y_train',shape=(n_train,),dtype='i1')
		outf.create_dataset('/y_test',shape=(n_test,),dtype='i1')
		# Go through file and write
		for i,(x,y) in enumerate(seqs_encode):
			if i < n_train:
				outf['/x_train'][i,...] = x
				outf['/y_train'][i] = y
			else:
				outf['/x_test'][i-n_train,...] = x
				outf['/y_test'][i-n_train] = y



def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-i1','--infile1',action='store',help='input file 1')
	parser.add_argument('-i2','--infile2',action='store',help='input file 2')
	parser.add_argument('-o','--outfile',action='store',help='output file')
	args = parser.parse_args()
	
	print('Labelling binding sequences...')
	
	label(args.infile1, args.infile1 + '_lbl',1)

	print('Labelling control sequences...')
	
	label(args.infile2, args.infile2 + '_lbl',0)

	print('Merging binding and control sequences...')
	
	merge_fasta(args.infile1 + '_lbl',args.infile2 + '_lbl',args.infile1 + args.infile2 + '_merge')

	print('Writing encoded sequences to hdf5 format...')

	to_hdf5(args.infile1 + args.infile2 + '_merge',args.outfile)

if __name__ == '__main__':
  main()