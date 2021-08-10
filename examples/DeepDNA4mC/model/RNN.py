from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Reshape
from keras.layers import LSTM, Bidirectional
from keras import optimizers


lstm_output_size = 128


print('Building model...')
model = Sequential()
model.add(Reshape((41,4),input_shape=(164,)))
print(model.outputs)
model.add(Bidirectional(LSTM(lstm_output_size)))
print(model.outputs)
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

