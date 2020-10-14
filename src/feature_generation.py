import numpy as np 
import pickle

def seq_to_vec(seq):
	conv_embed = {'A': 1, 'C': 2, 'G': 3, 'U': 4, 'N': 0, 'T': 4}
	embed = []
	for i in seq:
		embed.append(conv_embed[i])

	embed = np.asarray(embed)
	
	return embed

'''Motifs
0. DRACH (D:A/G/U, R:A/G, H=A/C/U) - Mammalian
1. RAC - Yeast
2. RRACH 
3. AAC
4. UAC(m6a)GAGAA
5. GAC
6. RRACU
'''

motifs = ['DRACH', 'RAC', 'RRACH', 'AAC', 'UACAGAGAA', 'GAC', 'RRACU']
D = ['A', 'G', 'U']
R = ['A', 'G']
H = ['A', 'C', 'U']

def seq_to_motif_vec(seq):
	motif_vec = np.zeros([len(seq), len(motifs)])
	for i in range(len(seq)):
		if seq[i] == 'A':
			# motif 0: DRACH
			try:
				if (seq[i-2] in D) and (seq[i-1] in R) and (seq[i+1] == 'C') and (seq[i+2] in H):
					motif_vec[i][0] = 1 
			except:
				pass
			# motif 1: RAC
			try:
				if (seq[i-1] in R) and (seq[i+1] == 'C'):
					motif_vec[i][1] = 1 
			except:
				pass
			# motif 2: RRACH
			try:
				if (seq[i-2] in R) and (seq[i-1] in R) and (seq[i+1] == 'C') and (seq[i+2] in H):
					motif_vec[i][2] = 1 
			except:
				pass
			# motif 3: AAC
			try:
				if (seq[i-1] == 'A') and (seq[i+1] == 'C'):
					motif_vec[i][3] = 1 
			except:
				pass
			# motif 4: UAC(m6a)GAGAA
			try:
				if (seq[i-3] == 'U') and (seq[i-2] == 'A') and (seq[i-1] == 'C') and (seq[i+1] == 'G') and (seq[i+2] == 'A') and (seq[i+3] == 'G') and (seq[i+4] == 'A') and (seq[i+5] == 'A'):
					motif_vec[i][4] = 1 
			except:
				pass
			# motif 5: GAC
			try:
				if (seq[i-1] == 'G') and (seq[i+1] == 'C'):
					motif_vec[i][5] = 1 
			except:
				pass
			# motif 6: RRACU
			try:
				if (seq[i-2] in R) and (seq[i-1] in R) and (seq[i+1] == 'C') and (seq[i+2] == 'U'):
					motif_vec[i][6] = 1 
			except:
				pass

	return motif_vec



# data preprocessing
# train
ds_dr_train = np.load('data_processed/dr_train.npy')
# print(ds_dr)

# matrix one-hot encoding
# filepath = 'dna2vec/pretrained/dna2vec-20161219-0153-k3to8-100d-10c-29320Mbp-sliding-Xat.w2v'
# mk_model = MultiKModel(filepath)

X_train = []
for i in range(1):
	print(ds_dr_train[i][0])
	seq_vec = seq_to_vec(ds_dr_train[i][0])
	motif_vec = seq_to_motif_vec(ds_dr_train[i][0])
	seq_vec = np.asarray(seq_vec)
	seq_vec = np.reshape(seq_vec, (101, 1))
	motif_vec = np.asarray(motif_vec)
	print(seq_vec.shape)
	print(motif_vec)
	print(motif_vec.shape)
	feature_vec = np.concatenate((seq_vec, motif_vec), axis = 1)
	print(feature_vec)

	X_train.append(feature_vec)

X_train = np.asarray(X_train)
print(X_train.shape)

# filename = 'X_test_dr.pickle'
# outfile = open(filename, 'wb')
# pickle.dump(X_train ,outfile)
# outfile.close()


