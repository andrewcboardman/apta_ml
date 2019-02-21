from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet.IUPAC import unambiguous_dna
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--outfile', type=str, action='store', dest='outfile', help='Output FASTAfile')
parser.add_argument('-n', '--n_seqs', type=int, action='store', default=1000,help='Number of random sequences to generate')
parser.add_argument('-L', '--length', type=int, action='store', default=40,help='Length of sequence')
args = parser.parse_args()

bases = ['A','C','G','T']
rand_bases = (np.random.choice(bases,size=args.length) for i in range(args.n_seqs))
rand_strings = (''.join(x) for x in rand_bases)
rand_seqs = (Seq(x,unambiguous_dna) for x in rand_strings)
rand_recs = (SeqRecord(seq,id='Random sequence {}'.format(i)) for (i,seq) in enumerate(rand_seqs))
SeqIO.write(rand_recs,args.outfile,'fasta')