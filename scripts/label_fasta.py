from Bio import SeqIO
import argparse
def lab(rec, label):
  rec.id = rec.id + label
  return rec

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-i','--infile',action='store',help='input file')
  parser.add_argument('-o','--outfile',action='store',help='output file')
  parser.add_argument('-l','--label',action='store',help='tag for ID')
  args = parser.parse_args()
  # load data from FASTA files
  seqs = SeqIO.parse(args.infile,'fasta')
  # label 
  seqs_l = (lab(rec,args.label) for rec in seqs)
  # write
  SeqIO.write(seqs_l,args.outfile,'fasta')
if __name__ == '__main__':
  main()



