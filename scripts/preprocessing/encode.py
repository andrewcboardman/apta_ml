from Bio import SeqIO
import argparse
import numpy as np


def vec_translate(a,alphabet):
  return np.vectorize(alphabet.__getitem__)(a)

def encode(rec,alphabet):
  seq = str(rec.seq)
  label = str(rec.id)
  chars = np.array(list(seq))
  code = vec_translate(chars,alphabet)
  return ','.join(code.astype('U1')) + ','+label +'\n'

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-i','--infile',action='store',help='input file')
  parser.add_argument('-o','--outfile',action='store',help='output file')
  parser.add_argument('-a','--alphabet',action='store',default='ACGT',help='alphabet to use')
  args = parser.parse_args()

  # load data from FASTA files
  seqs = SeqIO.parse(args.infile,'fasta')
  # label 
  seqs_encode = (encode(rec,args.alphabet) for rec in seqs)
  # write to file
  with open(args.outfile,'w') as fh:
    for code in seqs_encode:
      fh.write(code)
  
if __name__ == '__main__':
  main()