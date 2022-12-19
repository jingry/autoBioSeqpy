from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Reshape, GRU, Bidirectional, Flatten
from keras.layers import LSTM, Bidirectional
from keras import optimizers


lstm_output_size = 64
hidden_dims = 256


print('Building model...')
model = Sequential()
model.add(Reshape((41,4),input_shape=(164,)))
print(model.outputs)
model.add(Bidirectional(GRU(lstm_output_size, return_sequences=True)))
model.add(Flatten())
model.add(Dense(hidden_dims))
print(model.outputs)
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))
