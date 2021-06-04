from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Reshape
from keras.layers import Embedding
from keras.layers import LSTM, Bidirectional
from keras.layers import Conv1D, MaxPooling1D
from keras import optimizers



kernel_size = 9
filters = 250
pool_size = 10
lstm_output_size = 16




model = Sequential()
model.add(Reshape((2000,20),input_shape=(40000,)))
model.add(Dropout(0.2))
model.add(Conv1D(filters, kernel_size,padding ='valid',activation = 'relu',strides = 1))
model.add(MaxPooling1D(pool_size = pool_size))
model.add(Bidirectional(LSTM(lstm_output_size)))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss = 'binary_crossentropy',optimizer = optimizers.Adam(),metrics = ['acc'])



