from Bio import SeqIO

samples = [x for x in SeqIO.parse('MCMC_Samples.fa','fasta')]
pool = SeqIO.parse('hoinka_samples.fasta','fasta')
in_pool = [False] * len(samples)
for i,sample in enumerate(samples):
	for pool_rec in pool:
		if sample.seq == pool_rec.seq:
			in_pool[i] = True
			break
print(sum(in_pool))
