# Squeezenet Architecture (Papers 17)
# add dropout and regularization later on if needed

# Libraries
from keras.layers import TimeDistributed, Dense, Dropout, Input, LSTM, SpatialDropout1D, Embedding, Conv1D, Flatten
import keras
from keras.models import Model
import keras.backend as K 
from keras_self_attention import SeqSelfAttention

def NN():
	inp = Input(shape = (14,1,))
	
	x = Conv1D(128, 1, activation = 'relu')(inp)
	x = Conv1D(128, 1, activation = 'relu')(x)
	x = Conv1D(128, 1, activation = 'relu')(x)

	x = Flatten()(x)

	x = Dense(512, activation = 'relu')(x)
	x = Dense(256, activation = 'relu')(x)
	x = Dense(128, activation = 'relu')(x)
	x = Dense(64, activation = 'relu')(x)
	x = Dense(32, activation = 'relu')(x)

	out = Dense(2, activation = 'softmax')(x)

	model = Model(inputs = inp, outputs = out)
	print(model.summary())

	return model

