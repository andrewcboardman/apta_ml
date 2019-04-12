import itertools as it
import numpy as np
import argparse

class WordExtractor():
	def __init__(self,k):
		# length of word
		self.k = k 
		# Combinations of bases
		bases = [['A','C','G','T']]*self.k
		# Generate words
		self.words = np.array([''.join(letters) for letters in it.product(*bases)])
		
	def BagOfWords(self,strings):
		"""Test for the presence of every word in every string"""
		return np.core.defchararray.find(strings[:,np.newaxis],self.words[np.newaxis,:])

def main():
	parser = argparse.ArgumentParser()
    parser.add_argument('-i','--infile',action='store',help='Input file')
    parser.add_argument('-o','--outfile',action='store',help='Output file')
    parser.add_argument('-k',action='store',help='Length of words to be extracted')
    args = parser.parse_args()

    # Count reads so we can allocate memory
    reads = SeqIO.parse(args.infile,'fasta')
    L_read = len(next(reads).seq)
    n_reads = sum((1 for read in reads)) + 1

    # Get reads into numpy array for vectorised kmer counting
    reads = SeqIO.parse(args.infile,'fasta')
    reads_arr = np.empty((n_reads),dtype='U{}'.format(L_read))
    for i, read in enumerate(reads):
    	reads_arr[i] = str(read.seq.seq)

    # Extract word counts in strings
    xtr = WordExtractor(args.k)
    counts = xtr.BagOfWords(reads_arr)

    # Save counts to file
    np.savetxt(args.outfile,counts)






