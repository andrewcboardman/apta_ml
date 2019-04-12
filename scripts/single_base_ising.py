from Bio import SeqIO
from keras import to_categorical

base_dict = {'A':0,'C':1,'G':2,'T':3}


def encode(seq,mydict):
	seq = str(rec.seq)
	chars = np.array(list(seq))
	code = np.vectorize(mydict.__getitem__)(chars)
	return to_categorical(code,num_classes=4,dtype='int8')

reads = SeqIO.parse(infile)
one_base_encodings = (encode(read,base_dict) for read in reads)
one_two_base_encodings = ((code,np.outer(code.flatten(),code.flatten())) for code in one_base_encodings)

position_sums = 0
n_reads = 0
for encoded_read in encoded_reads:
	n_reads += 1
	position_sums += encoded_read
	n_reads += 1

np.savetxt(position_sums / n_reads)
np.save




