from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, Reshape
from keras.layers import Conv1D, MaxPooling1D
from keras import optimizers


filters = 250
kernel_size = 3
pool_size = 2
hidden_dims = 256

print('Building model...')
model = Sequential()
model.add(Reshape((41,4),input_shape=(164,)))
model.add(Conv1D(filters,kernel_size,padding = 'valid',activation = 'relu',strides = 1))
model.add(MaxPooling1D(pool_size = pool_size))
model.add(Flatten())
model.add(Dense(hidden_dims))
model.add(Dropout(0.3))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))
