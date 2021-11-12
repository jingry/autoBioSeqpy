from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Reshape
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import LSTM, Bidirectional
from keras import optimizers



filters = 150
kernel_size = 3
pool_size = 2
lstm_output_size = 128

print('Building model...')
model = Sequential()
model.add(Reshape((41,4),input_shape=(164,)))
model.add(Conv1D(filters,kernel_size,padding = 'valid',activation = 'relu',strides = 1))
model.add(MaxPooling1D(pool_size = pool_size))
model.add(Bidirectional(LSTM(lstm_output_size)))
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

