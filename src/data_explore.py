from Bio import SeqIO
import numpy as np 
import pickle

# pos_file = '../repos/DeepM6ASeq/data/data/dr/test_pos.fa'

# ds = []
# for record in SeqIO.parse(pos_file, "fasta"):
#     m6asite = int(record.id.split(';')[4].split('::')[0])
#     seq = str(record.seq)
#     if seq[m6asite] == 'A':
#     	arr = [seq, m6asite]
#     	ds.append(arr)
    
# length_pos = len(ds)

# neg_file = '../repos/DeepM6ASeq/data/data/dr/test_neg.fa'

# for record in SeqIO.parse(neg_file, "fasta"):
# 	m6asite = -1
# 	seq = str(record.seq)
# 	arr = [seq, m6asite]
# 	ds.append(arr)
    
# length_neg = len(ds) - length_pos

# print(length_pos, length_neg)

# ds = np.asarray(ds)
# np.save('dr_test', ds)

# motif analysis
# ds_dr_train = np.load('data_processed/dr_train.npy')
# infile = open('X_train_dr.pickle','rb')
# X_train = pickle.load(infile)
# infile.close()

# # sequence tag for m6A
# y_train = []
# for i in range(len(ds_dr_train)):
# 	y = np.zeros((101))
# 	m6asite = int(ds_dr_train[i][1])
# 	if m6asite != -1:
# 		y[m6asite] = 1
# 	y_train.append(y)

# # print(y_train, X_train)
# motifs = np.zeros([7])
# count = 0
# motifs_not = np.zeros([7])
# count_not = 0
# for i in range(len(y_train)):
# 	for j in range(101):
# 		if y_train[i][j] == 1.0:
# 			count += 1
# 			print(y_train[i][j], X_train[i][j][1:])
# 			motifs += X_train[i][j][1:]
# 		else:
# 			if X_train[i][j][0] == 1:
# 				count_not += 1
# 				motifs_not += X_train[i][j][1:]

# print("M6A present: ", motifs/count)
# print("M6A not present, just a basic Adenine: ", motifs_not/count_not)

'''
# Motifs: ['DRACH', 'RAC', 'RRACH', 'AAC', 'UACAGAGAA', 'GAC', 'RRACU']
# M6A present: [0.720076, 0.98507193, 0.720076, 0.57749027, 0., 0.40758165, 0.]
# M6A not present, just a basic Adenine: [0.03192914, 0.09775058, 0.03192914, 0.05575539, 0., 0.04199518, 0.]
# Motifs are a good indicator. 
'''

# take each individual adenine and make test cases.
ds_dr_train = np.load('data_processed/dr_train.npy')
infile = open('X_train_dr.pickle','rb')
X_train = pickle.load(infile)
infile.close()

# sequence tag for m6A
y_train = []
for i in range(len(ds_dr_train)):
	y = np.zeros((101))
	m6asite = int(ds_dr_train[i][1])
	if m6asite != -1:
		y[m6asite] = 1
	y_train.append(y)

X_t = []
y_t = []

for i in range(len(y_train)):
	print(i, len(y_train))
	for j in range(len(y_train[i])):
		if X_train[i][j][0] == 1:
			y_t.append(y_train[i][j])
			X_t.append(X_train[i][j][1:])
		# print(X_train[i][j][1:], y_train[i][j])

X_t = np.asarray(X_t)
y_t = np.asarray(y_t)
print(X_t.shape, y_t.shape)

filename = 'X_train_dr_one.pickle'
outfile = open(filename, 'wb')
pickle.dump(X_t ,outfile)
outfile.close()

filename = 'y_train_dr_one.pickle'
outfile = open(filename, 'wb')
pickle.dump(y_t ,outfile)
outfile.close()

# 80-20 split