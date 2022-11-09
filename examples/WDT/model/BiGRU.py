from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation,Reshape, GRU, Bidirectional, Flatten
from keras.utils import np_utils
from keras import optimizers

lstm_output_size = 16
hidden_dims = 650




print('Building model...')
model = Sequential()
model.add(Reshape((350,74),input_shape=(25900,)))
model.add(Bidirectional(GRU(lstm_output_size, return_sequences=True)))
model.add(Flatten())
model.add(Dense(hidden_dims))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))



