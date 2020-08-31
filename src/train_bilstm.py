'''
Model training with the entire training data
'''

# Libraries
import pandas as pd
import numpy as np
import keras
import tensorflow as tf
from keras.models import Model
from tensorflow.keras.models import load_model
import keras.backend as K
from keras import optimizers
from keras.layers import Dense, Dropout, BatchNormalization, Conv1D, Flatten, Input, GaussianNoise, LeakyReLU, Add
from keras.utils import to_categorical, np_utils
from keras.regularizers import l2
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, normalize
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import pickle
from keras import regularizers
from keras import backend as K
from sklearn.utils import class_weight
from model_bilstm import BiLSTM


# GPU
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

tf.keras.backend.clear_session()



config = ConfigProto()
config.gpu_options.allow_growth = True
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

LIMIT = 3 * 1024
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=LIMIT)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

def sensitivity(y_true, y_pred):
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
	
	return true_positives / (possible_positives + K.epsilon())

def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))

    return true_negatives / (possible_negatives + K.epsilon())

def create_weighted_binary_crossentropy(zero_weight, one_weight):

    def weighted_binary_crossentropy(y_true, y_pred):
    	b_ce = K.binary_crossentropy(y_true, y_pred)
    	weight_vector = y_true * one_weight + (1. - y_true) * zero_weight
    	weighted_b_ce = weight_vector * b_ce

    	return K.mean(weighted_b_ce)

    return weighted_binary_crossentropy

def seq_to_mat(seq):
	conv_embed = {'A': 1, 'C': 2, 'G': 3, 'U': 4, 'N': 0, 'T': 4}
	embed = []
	for i in seq:
		embed.append(conv_embed[i])

	embed = np.asarray(embed)
	
	return embed

# data preprocessing
# train
ds_dr_train = np.load('dr_train.npy')
# print(ds_dr)

# matrix one-hot encoding
# filepath = 'dna2vec/pretrained/dna2vec-20161219-0153-k3to8-100d-10c-29320Mbp-sliding-Xat.w2v'
# mk_model = MultiKModel(filepath)

X_train = []
for i in range(len(ds_dr_train)):
	mat_embed = seq_to_mat(ds_dr_train[i][0])
	# vec = mk_model.vector(ds_dr_train[i][0])
	# print(vec, vec.shape)
	# mat_embed = ds_dr_train[i][0]
	X_train.append(mat_embed)

# print(X_train)

# sequence tag for m6A
y_train = []
for i in range(len(ds_dr_train)):
	y = np.zeros((101))
	m6asite = int(ds_dr_train[i][1])
	if m6asite != -1:
		y[m6asite] = 1
	y_train.append(y)

# test
ds_dr_test = np.load('dr_test.npy')
# print(ds_dr)

# matrix one-hot encoding
X_test = []
for i in range(len(ds_dr_test)):
	mat_embed = seq_to_mat(ds_dr_test[i][0])
	# mat_embed = ds_dr_test[i][0]
	# print(i, mat_embed.shape)
	X_test.append(mat_embed)

# print(X_test)

# sequence tag for m6A
y_test = []
for i in range(len(ds_dr_test)):
	y = np.zeros((101))
	m6asite = int(ds_dr_test[i][1])
	if m6asite != -1:
		y[m6asite] = 1
	y_test.append(y)

# print(y_train[0])
X_train = np.asarray(X_train)
# X_train = np.expand_dims(X_train, axis=2)
# print(X_train[0], y_train[0], ds_dr_train[0][0], ds_dr_train[0][1])
X_test = np.asarray(X_test)
# X_test = np.expand_dims(X_test, axis=2)
print(X_test.shape)

y_train = np.asarray(y_train)
y_test = np.asarray(y_test)
y_train = np.expand_dims(y_train, axis=2)
y_test = np.expand_dims(y_test, axis=2)
X_train, y_train = shuffle(X_train, y_train, random_state = 42)

weighted_loss = create_weighted_binary_crossentropy(zero_weight = 1, one_weight = 201)

# CNN Model
model = BiLSTM()
opt = keras.optimizers.SGD(learning_rate = 1e-5, momentum = 0.9)
model.compile(optimizer = 'adam', loss = weighted_loss, metrics = [sensitivity, 'accuracy'])

mcp_save = keras.callbacks.callbacks.ModelCheckpoint('bilstm_matrixembed_dr.h5', save_best_only=True, monitor='val_loss', verbose=1)
reduce_lr = keras.callbacks.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
callbacks_list = [mcp_save, reduce_lr]
# # with tf.device('/cpu:0'):
# history = model.fit(X_train, y_train, batch_size = 256, epochs = 500, validation_data = (X_test, y_test), shuffle = False, callbacks = callbacks_list)
# model.save('bilstm_matrixembed.h5')

# # checking
model = load_model('bilstm_matrixembed_dr.h5', custom_objects = {'weighted_binary_crossentropy': weighted_loss, 'sensitivity': sensitivity})

pred = model.predict(X_test)

print(pred.shape)
# for i in range(len(y_test)):
# 	X = X_test[i]
# 	X = np.expand_dims(X, axis=0)
# 	pred = model.predict(X)
# 	pred = np.asarray(pred)
# 	pred = np.squeeze(pred, axis=0)
# 	pred = np.squeeze(pred, axis=1)
# 	print(pred.shape)
# 	if ds_dr_test[i][1] != -1:
# 		print("Yes m6A Site: ")
# 		ind = ds_dr_test[i][1]
# 		print(ind)
# 		print(ind, pred)
# 	else:
# 		print("Not m6A Site: ")
# 		print(ds_dr_test[i][1], max(pred[0]))

'''Performace (Sensitivity)
Matrix Embeddings: 52.16%
DNA2Vec Embedding:
Keras Embedding:
'''