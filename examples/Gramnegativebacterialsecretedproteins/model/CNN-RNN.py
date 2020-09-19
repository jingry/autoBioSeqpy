from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Reshape
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import LSTM, Bidirectional
from keras.utils import np_utils
from keras import optimizers


filters = 250
kernel_size = 5
pool_size = 2
lstm_output_size = 64

print('Building model...')
model = Sequential()
model.add(Reshape((2000,20),input_shape=(40000,)))
model.add(Conv1D(filters,kernel_size,padding = 'valid',activation = 'relu',strides = 1))
model.add(MaxPooling1D(pool_size = pool_size))
model.add(Bidirectional(LSTM(lstm_output_size)))
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(6))
model.add(Activation('softmax'))
model.compile(loss = 'categorical_crossentropy',optimizer = optimizers.Adam(),metrics = ['acc'])

