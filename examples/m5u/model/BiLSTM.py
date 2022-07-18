from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Reshape, LSTM, Bidirectional, Flatten
from keras import optimizers


lstm_output_size = 256
hidden_dims = 256


print('Building model...')
model = Sequential()
model.add(Reshape((41,4),input_shape=(164,)))
model.add(Bidirectional(LSTM(lstm_output_size, return_sequences=True)))
model.add(Bidirectional(LSTM(lstm_output_size, return_sequences=True)))
model.add(Flatten())
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

