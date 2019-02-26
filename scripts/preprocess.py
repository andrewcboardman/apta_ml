from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import generic_dna
from Bio import SeqIO
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-f','--forward',action='store',type=str,help='Input file for forward reads')
parser.add_argument('-r','--reverse',action='store',type=str,help='Input file for reverse reads')
parser.add_argument('-o','--output',action='store',type=str,help='output file')
args = parser.parse_args()

def CheckSeq(seq):
	if len(seq.seq) == 40 and min(seq.letter_annotations["phred_quality"]) >= 20 and 'N' not in seq.seq:
		return True

def ProcessPair(fwd_seq,rev_seq):
	if CheckSeq(fwd_seq):
		return fwd_seq
	elif CheckSeq(rev_seq):
		return SeqRecord(rev_seq.seq.reverse_complement(),id=rev_seq.id)
	else:
		return SeqRecord(Seq(''),id='none')

fwd = SeqIO.parse(args.forward,'fastq')
rev = SeqIO.parse(args.reverse,'fastq')

apta_seqs = (ProcessPair(fwd_seq,rev_seq) for (fwd_seq,rev_seq) in zip(fwd,rev))
apta_seqs_qc = (seq for seq in apta_seqs if seq.id != None)
SeqIO.write(apta_seqs_qc,args.output,'fasta')
