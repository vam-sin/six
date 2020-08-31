# Squeezenet Architecture (Papers 17)
# add dropout and regularization later on if needed

# Libraries
from keras.layers import TimeDistributed, Dense, Dropout, Input, Bidirectional, LSTM, SpatialDropout1D, Embedding
import keras
from keras.models import Model
import keras.backend as K 

def BiLSTM():
	inp = Input(shape = (101,))
	embed = Embedding(101, 128)(inp)

	bid = Bidirectional(LSTM(32, return_sequences=True))(embed)
	bid = SpatialDropout1D(0.5)(bid)
	# bid = Bidirectional(LSTM(256, return_sequences=True))(bid)
	# bid = SpatialDropout1D(0.9)(bid)
	out = TimeDistributed(Dense(1, activation = 'sigmoid'))(bid)

	model = Model(inputs = inp, outputs = out)
	print(model.summary())

	return model

