'''
Model training with the entire training data
'''
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

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
from model_nn import NN
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

# data preprocessing
# training set
infile = open('../features/Motifs/dr_train.pickle','rb')
X_train_motif = pickle.load(infile)
infile.close()

infile = open('../features/SSF/dr_train_3.pickle','rb')
X_train_ss = pickle.load(infile)
infile.close()

infile = open('../features/Labels/dr_train.pickle','rb')
y_train = pickle.load(infile)
infile.close()

# testing set
infile = open('../features/Motifs/dr_test.pickle','rb')
X_test_motif = pickle.load(infile)
infile.close()

infile = open('../features/SSF/dr_test_3.pickle','rb')
X_test_ss = pickle.load(infile)
infile.close()

infile = open('../features/Labels/dr_test.pickle','rb')
y_test = pickle.load(infile)
infile.close()

# Making dataset
X_train = np.concatenate([X_train_motif, X_train_ss], axis=1)
X_test = np.concatenate([X_test_motif, X_test_ss], axis=1)
# print(X_train.shape, X_test.shape)

X_train = np.asarray(X_train)
X_test = np.asarray(X_test)
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)

# sample dataset with undersampling
print("Resampling")
undersample = NearMiss(version=1, n_neighbors=5)
X_train, y_train = undersample.fit_resample(X_train, y_train)
X_test_sample, y_test_sample = undersample.fit_resample(X_test, y_test)
print("Completed Resampling")

# expand dims
X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)
X_test_sample = np.expand_dims(X_test_sample, axis=2)
X_train, y_train = shuffle(X_train, y_train, random_state = 42)

# CNN Model
model = NN()
opt = keras.optimizers.Adam(learning_rate = 1e-5)
model.compile(optimizer = "adam", loss = 'categorical_crossentropy', metrics = ['accuracy'])

# # weights
# weights = class_weight.compute_class_weight('balanced',
#                                             np.unique(y_train),
#                                             y_train)

# print(weights)
weights = [1, 0.000001]
# weights = [50248756, 0.1010]
# weights = [1,1]
# print(weights)
# print(np.unique(y_train, return_counts = True))
# print(np.unique(y_test, return_counts = True))

# y processing
y_train = to_categorical(y_train, num_classes = 2)
y_test = to_categorical(y_test, num_classes = 2)
y_test_sample = to_categorical(y_test_sample, num_classes = 2)

# checkpoints
mcp_save = keras.callbacks.callbacks.ModelCheckpoint('../models/nn.h5', save_best_only=True, monitor='val_accuracy', verbose=1)
reduce_lr = keras.callbacks.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=10, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
callbacks_list = [mcp_save, reduce_lr]

with tf.device('/gpu:0'):
  # history = model.fit(X_train, y_train, batch_size = 256, epochs = 200, validation_data = (X_test_sample, y_test_sample), shuffle = False, callbacks = callbacks_list)

  model = load_model('../models/nn.h5')

  # sample test
  eval_ = model.evaluate(x = X_test_sample, y = y_test_sample)
  print("Loss Sample: " + str(eval_[0]) + ", Accuracy Sample: " + str(eval_[1]))
  print(eval_)

  y_pred = model.predict(X_test_sample)

  # Metrics
  print("Confusion Matrix Sample")
  matrix = confusion_matrix(y_test_sample.argmax(axis=1), y_pred.argmax(axis=1))
  print(classification_report(y_test_sample.argmax(axis=1), y_pred.argmax(axis=1)))
  print(matrix)

  eval_ = model.evaluate(x = X_test, y = y_test)
  print("Loss: " + str(eval_[0]) + ", Accuracy: " + str(eval_[1]))
  print(eval_)

  y_pred = model.predict(X_test)

  # Metrics
  print("Confusion Matrix")
  matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
  print(classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1)))
  print(matrix)

''' Results on full test dataset (DR) (Adding the SS Folds reducing false positives)
Loss: 0.30297964811325073, Accuracy: 0.8687190413475037
[0.30297964811325073, 0.8687190413475037]
Confusion Matrix
              precision    recall  f1-score   support

           0       0.91      0.82      0.86      2826
           1       0.84      0.92      0.87      2826

    accuracy                           0.87      5652
   macro avg       0.87      0.87      0.87      5652
weighted avg       0.87      0.87      0.87      5652

[[2318  508]
 [ 234 2592]]
'''