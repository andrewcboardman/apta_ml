######### step 1: Labelling and combining ###################


from Bio import SeqIO

def label(rec, label):
	rec.id = rec.id + label
	return rec

def combine(infile1,outfile,label):
	# load data from FASTA files
	seqs = SeqIO.parse(infile,'fasta')
	# label 
	seqs_l = (label(rec,label) for rec in seqs)
	# write
	SeqIO.write(seqs1_l,outfile)



