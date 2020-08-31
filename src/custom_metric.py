import keras
import keras.backend as K
import numpy as np
import tensorflow as tf
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

y_pred = np.zeros((101))
y_true = np.zeros((101))
y_true[10] = 1
y_pred[10] = 1

def sensitivity(y_true, y_pred):
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
	
	return true_positives / (possible_positives + K.epsilon())

def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))

    return true_negatives / (possible_negatives + K.epsilon())

# def weighted_loss(y_true, y_pred):
# 	result = []
# 	L = len(y_pred)
# 	for i in range(L):
# 		y_pred[i] = [max(min(x, 1 - K.epsilon()), K.epsilon()) for x in y_pred[i]]
# 		result.append(-np.mean([y_true[i][j] * math.log(y_pred[i][j]) + (1 - y_true[i][j]) * math.log(1 - y_pred[i][j]) for j in range(len(y_pred[i]))]))

# 	return np.mean(result)

one_weight = 10
zero_weight = 1

def weighted_binary_crossentropy(y_true, y_pred):
    if K.sum(y_true) == 1.0:
    	b_ce = K.binary_crossentropy(y_true, y_pred)
    	print(b_ce)
    	weight_vector = y_true * one_weight + (1. - y_true) * zero_weight
    	print(weight_vector)
    	weighted_b_ce = weight_vector * b_ce
    	print(weighted_b_ce)
    	return K.mean(weighted_b_ce)
    else:
    	b_ce = K.binary_crossentropy(y_true, y_pred)
    	print(b_ce)
    	weight_vector = y_true * one_weight + (1. - y_true) * zero_weight
    	print(weight_vector)
    	weighted_b_ce = weight_vector * b_ce
    	print(weighted_b_ce)



print(y_pred, y_true)
y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
print(weighted_binary_crossentropy(y_true, y_pred))
print(sensitivity(y_true, y_pred))
print(specificity(y_true, y_pred))

'''
Sens = TP / (TP+FN) = 0
'''