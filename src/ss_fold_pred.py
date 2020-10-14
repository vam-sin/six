import sys
import numpy as np
import pickle

egg_path = '/usr/local/lib/python3.8/site-packages'

sys.path.append(egg_path)

import RNA

# train
ds_dr_train = np.load('data_processed/dr_test.npy')
# print(ds_dr)

# matrix one-hot encoding
# filepath = 'dna2vec/pretrained/dna2vec-20161219-0153-k3to8-100d-10c-29320Mbp-sliding-Xat.w2v'
# mk_model = MultiKModel(filepath)
window = 3
X_train = []
for i in range(len(ds_dr_train)):
	seq = ds_dr_train[i][0]
	print(i, len(ds_dr_train))
	(ss, mfe) = RNA.fold(seq)
	ss = str(ss)
	ss = ss.replace('(', '0')
	ss = ss.replace('.', '1')
	ss = ss.replace(')', '2')
	for j in range(len(seq)):
		if seq[j] == 'A':
			feat = ''
			for k in range(window*2 + 1):
				try:
					feat += ss[j + k - window]
				except:
					feat += '3'
			# print(feat)
			feat_vec = []
			for f in range(len(feat)):
				feat_vec.append(int(feat[f]))
			# print(feat_vec)
			X_train.append(feat_vec)

X_train = np.asarray(X_train)
print(X_train.shape)

filename = 'X_test_ssfold_3_dr.pickle'
outfile = open(filename, 'wb')
pickle.dump(X_train ,outfile)
outfile.close()