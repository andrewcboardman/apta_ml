from Bio import SeqIO
import argparse
import numpy as np

base_dict = {'A':0,'C':1,'G':2,'T':3}
def vec_translate(a,mydict):
  return np.vectorize(mydict.__getitem__)(a)

def encode(rec):
  seq = str(rec.seq)
  ix = str(rec.id)
  chars = np.array(list(seq))
  code = vec_translate(chars,base_dict)
  label = ix[-1]
  return ','.join(code.astype('U1')) + ','+label +'\n'

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-i','--infile',action='store',help='input file')
  parser.add_argument('-o','--outfile',action='store',help='output file')
  args = parser.parse_args()

  # load data from FASTA files
  seqs = SeqIO.parse(args.infile,'fasta')
  # label 
  seqs_encode = (encode(rec) for rec in seqs)
  # write to file
  with open(args.outfile,'w') as fh:
    for code in seqs_encode:
      fh.write(code)
  
if __name__ == '__main__':
  main()