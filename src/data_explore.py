from Bio import SeqIO
import numpy as np 

pos_file = '../repos/DeepM6ASeq/data/data/dr/test_pos.fa'

ds = []
for record in SeqIO.parse(pos_file, "fasta"):
    m6asite = int(record.id.split(';')[4].split('::')[0])
    seq = str(record.seq)
    if seq[m6asite] == 'A':
    	arr = [seq, m6asite]
    	ds.append(arr)
    
length_pos = len(ds)

neg_file = '../repos/DeepM6ASeq/data/data/dr/test_neg.fa'

for record in SeqIO.parse(neg_file, "fasta"):
	m6asite = -1
	seq = str(record.seq)
	arr = [seq, m6asite]
	ds.append(arr)
    
length_neg = len(ds) - length_pos

print(length_pos, length_neg)

ds = np.asarray(ds)
np.save('dr_test', ds)