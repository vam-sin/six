# libraries
# Libraries
import pandas as pd
import numpy as np
import keras
import math
from keras.models import Model
from tensorflow.keras.models import load_model
from keras.utils import to_categorical
import keras.backend as K
from keras import optimizers
from keras.layers import Dense, Dropout, BatchNormalization, Conv1D, Flatten, Input, GaussianNoise, LeakyReLU, Add
from keras.utils import to_categorical, np_utils
from keras.regularizers import l2
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, normalize
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt
import pickle
from keras import regularizers
from keras import backend as K
from sklearn.utils import class_weight
from model_bilstm import BiLSTM
import keras_self_attention
from sklearn.utils import class_weight
import pickle
from keras.utils import to_categorical 
from imblearn.under_sampling import NearMiss

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

infile = open('X_test_dr_one.pickle','rb')
X_test = pickle.load(infile)
infile.close()

infile = open('y_test_dr_one.pickle','rb')
y_test = pickle.load(infile)
infile.close()

prec = []
import math
for i in range(math.ceil(len(pred))):
	mask = []
	for j in ds_dr_test[i][0]:
		if j == 'A':
			mask.append(1)
		else:
			mask.append(0)
	sigmoid_pred = []
	for j in range(len(mask)):
		val = pred[i][j]
		sigmoid_pred.append(mask[j] * pred[i][j])
	sigmoid_pred = np.asarray(sigmoid_pred)
	m6asite = int(ds_dr_test[i][1])
	y_pred = []
	y_true = []
	# print(sigmoid_pred)
	for j in range(len(sigmoid_pred)):
		if sigmoid_pred[j] >= thresh:
			y_pred.append(1)
		else:
			y_pred.append(0)
	for j in range(len(sigmoid_pred)):
		if j == m6asite:
			y_true.append(1)
		else:
			y_true.append(0)

	cm = confusion_matrix(y_true, y_pred)
	# print(cm)
	try:
		precision = cm[1][1]/(cm[1][1] + cm[0][1] + np.finfo(float).eps)
	except:
		precision = 1.0
	prec.append(precision)
	print(np.mean(prec), precision)

print("Mean Precision: ", np.mean(prec))