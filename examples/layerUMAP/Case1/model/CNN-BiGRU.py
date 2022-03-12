from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Reshape
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.layers import GRU, Bidirectional
from keras.utils import np_utils
from keras import optimizers


filters = 300
kernel_size = 5
lstm_output_size = 64
hidden_dims = 650


print('Building model...')
model = Sequential()
model.add(Reshape((2000,20),input_shape=(40000,)))
model.add(Conv1D(filters,kernel_size,padding = 'valid',activation = 'relu',strides = 1))
model.add(Conv1D(filters,kernel_size,padding = 'valid',activation = 'relu',strides = 1))
model.add(Bidirectional(GRU(lstm_output_size, return_sequences=True)))
model.add(Bidirectional(GRU(lstm_output_size, return_sequences=True)))
model.add(Bidirectional(GRU(lstm_output_size, return_sequences=True)))
model.add(GlobalMaxPooling1D())
model.add(Dense(hidden_dims))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(6))
model.add(Activation('softmax'))

model.compile(loss = 'categorical_crossentropy',optimizer = optimizers.Adam(),metrics = ['acc'])
