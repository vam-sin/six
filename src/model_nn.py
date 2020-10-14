# Squeezenet Architecture (Papers 17)
# add dropout and regularization later on if needed

# Libraries
from keras.layers import TimeDistributed, Dense, Dropout, Input, Bidirectional, LSTM, SpatialDropout1D, Embedding, Conv1D, Flatten
import keras
from keras.models import Model
import keras.backend as K 
from keras_self_attention import SeqSelfAttention

def NN():
	inp = Input(shape = (14,1,))
	
	bid = Conv1D(128, 1, activation = 'relu')(inp)
	bid = Conv1D(128, 1, activation = 'relu')(bid)
	bid = Conv1D(128, 1, activation = 'relu')(bid)

	bid = Flatten()(bid)

	bid = Dense(512, activation = 'relu')(bid)
	bid = Dense(256, activation = 'relu')(bid)
	bid = Dense(128, activation = 'relu')(bid)
	bid = Dense(64, activation = 'relu')(bid)
	bid = Dense(32, activation = 'relu')(bid)

	out = Dense(2, activation = 'softmax')(bid)

	model = Model(inputs = inp, outputs = out)
	print(model.summary())

	return model

